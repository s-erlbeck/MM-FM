# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import os
import torch
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging

import math
import gc
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler, VAEPriorSampler, sample_prior_noise
from stage2.transport.gmm_sampler import GMMSampler
from stage2.transport.adaptive_weighter import AdaptiveClassWeighter
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config, is_model_conditional
from utils.data_utils import ClassBalancedSubset, ImageNetDataset
from utils import wandb_utils
from utils.optim_utils import build_optimizer, build_scheduler
from utils.eval_utils import evaluate_fid
import warnings

# Suppress PyTorch autocast deprecation warning that floods the log (still works fine, just old API)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        gmm_config,
        fid_config,
        data_limit_config,
        data_config,
        vae_prior_config,
    ) = parse_configs(args.config)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)
    gmm_cfg = to_dict(gmm_config)
    fid_cfg = to_dict(fid_config)
    data_limit_cfg = to_dict(data_limit_config)
    data_cfg = to_dict(data_config)
    vae_prior_cfg = to_dict(vae_prior_config)

    # Validate --data-path is provided (ImageNetDataset is the only supported data source)
    if args.data_path is None:
        raise ValueError("--data-path is required")

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))

    # Initial phase configuration (optional)
    initial_phase_epochs = int(training_cfg.get("initial_phase_epochs", 0))
    initial_ema_decay = float(training_cfg.get("initial_ema_decay", ema_decay))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    sample_every = int(training_cfg.get("sample_every", 10_000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    # FID evaluation configuration
    fid_enabled = fid_cfg.get("enabled", False)
    fid_every = int(fid_cfg.get("eval_every", 5000))
    fid_num_samples = int(fid_cfg.get("num_samples", 5000))
    fid_batch_size = int(fid_cfg.get("batch_size", 32))
    fid_ref_batch = fid_cfg.get("ref_batch_path", None)

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the RAE encoder.")

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*")) - 1
        model_target = str(model_config.get("target", "stage2"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        loss_weight_str = loss_weight if loss_weight is not None else "none"
        experiment_name = (
            f"{os.path.basename(args.results_dir)}/{experiment_index:03d}-{model_string_name}-"
            f"{path_type}-{prediction}-{loss_weight_str}{precision_suffix}-acc{grad_accum_steps}"
        )
        experiment_dir = os.path.join(args.results_dir, experiment_name.split("/")[-1])
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = "MM-FM"
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    opt_state = None
    sched_state = None
    train_steps = 0
    start_epoch = 0

    # Load checkpoint if resuming training
    weighter_state_path = None
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")  # Loaded later (see DEFERRED OPTIMIZER STATE LOADING)
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))
        weighter_state_path = checkpoint.get("weighter_path")
        # Resume from next epoch (backward compatible with checkpoints without "epoch" field)
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1

        # Debug logging for checkpoint contents
        if rank == 0:
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Free checkpoint dict to reduce memory pressure
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"Checkpoint loaded. Resuming from step {train_steps}, epoch {start_epoch}")
            logger.info(f"GPU memory after checkpoint load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    # For unconditional models, y_embedder is unused so we need find_unused_parameters=True
    find_unused = not is_model_conditional(model)
    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False, find_unused_parameters=find_unused)

    # Memory cleanup after DDP wrapping
    gc.collect()
    torch.cuda.empty_cache()

    # Collect all trainable parameters:
    # - DiT model (via DDP)
    trainable_params = list(model.parameters())
    opt, opt_msg = build_optimizer(trainable_params, training_cfg)

    # ==========================================================================
    # DEFERRED OPTIMIZER STATE LOADING
    # ==========================================================================
    # When resuming from checkpoint, we defer loading the optimizer state until
    # after the first forward/backward pass. This avoids OOM errors caused by
    # memory allocation order:
    #
    # Problem: AdamW optimizer stores momentum + variance buffers (2x model size,
    # ~6.8GB for 847M params). When loaded upfront during resume, these buffers
    # claim GPU memory before activation tensors are allocated, causing OOM
    # during the first forward pass.
    #
    # Solution: Let the first step run with fresh optimizer (no momentum), which
    # allows PyTorch to allocate activation memory first. Then load the saved
    # optimizer state, which overwrites the fresh buffers with the checkpointed
    # momentum/variance values.
    #
    # Trade-off: The first step after resume has a slightly different weight
    # update (like SGD without momentum history). However:
    #   - The momentum/variance history is fully restored after this step
    #     (load_state_dict overwrites the fresh buffers)
    #   - Only 1 weight update out of 80,000+ is affected
    #   - Impact on final model quality is negligible
    #
    # Alternative (PyTorch 2.x+): Load optimizer state BEFORE opt.step() but
    # AFTER backward(). PyTorch 2.x's load_state_dict() auto-casts CPU tensors
    # to GPU via _process_value_according_to_param_policy(). This skips no steps
    # but requires PyTorch 2.x. Flow: forward() -> backward() -> load_state_dict()
    # -> opt.step(). Current implementation is more conservative for compatibility.
    # ==========================================================================
    deferred_opt_state = opt_state
    opt_state = None

    # ========================================================================
    # DATA LOADING: ImageNetDataset (reads directly from a train/val.zip)
    # ========================================================================
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ImageNetDataset(args.data_path, transform=transform)

    # Apply data limiting if configured (before DistributedSampler)
    data_limit_enabled = data_limit_cfg.get('enabled', False)
    if data_limit_enabled:
        sample_percentage = data_limit_cfg.get('sample_percentage', 1.0)
        class_percentage = data_limit_cfg.get('class_percentage', 1.0)
        seed = data_limit_cfg.get('seed', 42)

        original_size = len(dataset)
        logger.info(
            f"Applying data limit: {sample_percentage*100:.1f}% of samples from "
            f"{class_percentage*100:.1f}% of classes (seed={seed})"
        )
        dataset = ClassBalancedSubset(
            dataset,
            sample_percentage=sample_percentage,
            class_percentage=class_percentage,
            seed=seed,
        )
        logger.info(f"Dataset reduced from {original_size:,} to {len(dataset):,} samples")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Log dataset size (keep existing logging for non-limited case)
    if not data_limit_enabled:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    steps_per_epoch = loader_batches // grad_accum_steps

    # Warn if there are incomplete batches that will be dropped
    if loader_batches % grad_accum_steps != 0:
        incomplete_batches = loader_batches % grad_accum_steps
        if rank == 0:
            logger.warning(
                f"Gradient accumulation mismatch: {loader_batches} batches with grad_accum_steps={grad_accum_steps}. "
                f"Dropping {incomplete_batches} incomplete batch(es) per epoch (negligible with shuffle)."
            )
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    # Free scheduler state after loading
    if sched_state is not None:
        del sched_state
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)
        if initial_phase_epochs > 0:
            initial_lr = float(training_cfg.get("initial_lr", training_cfg.get("base_lr", 2e-4)))
            logger.info(f"Initial phase: epochs 0-{initial_phase_epochs} with lr={initial_lr}, ema_decay={initial_ema_decay}")

    # Log FID evaluation configuration
    if fid_enabled:
        if fid_ref_batch is None:
            raise ValueError("FID evaluation is enabled but ref_batch_path is not provided in config.")
        if rank == 0:
            logger.info(f"FID evaluation enabled: evaluating every {fid_every} steps")
            logger.info(f"  Reference batch: {fid_ref_batch}")
            logger.info(f"  Number of samples: {fid_num_samples}")
            logger.info(f"  Batch size per GPU: {fid_batch_size}")
            logger.info(f"  Distributed sampling: {world_size} GPUs (each generates ~{int(np.ceil(fid_num_samples/world_size))} samples)")

    # Initialize GMM sampler if enabled
    gmm_sampler = None
    use_mode_conditional = gmm_cfg.get('mode_conditional', False)
    use_gmm = gmm_cfg.get('enabled', False)

    if use_gmm:
        gmm_path = gmm_cfg.get('path')
        if gmm_path is None:
            raise ValueError("GMM path is required when gmm.enabled=true")

        # Load CLS GMM for mode assignment if enabled
        # Needed for both mode-conditional AND unconditional GMM (for mode prediction)
        cls_gmm_path = None
        cls_enabled = gmm_cfg.get('cls_enabled', False)
        if cls_enabled:
            cls_gmm_path = gmm_cfg.get('cls_path')
            if cls_gmm_path is None:
                raise ValueError("CLS GMM path is required when gmm.cls_enabled=true")

        # Validate: mode_conditional requires cls_enabled
        if use_mode_conditional and not cls_enabled:
            raise ValueError("gmm.cls_enabled must be true when gmm.mode_conditional=true")

        gmm_sampler = GMMSampler(
            gmm_path=gmm_path,
            cls_gmm_path_for_assignment=cls_gmm_path,
            cls_assignment_mode=gmm_cfg.get('cls_assignment_mode', 'soft'),
            verbose=(rank == 0),
            validate_gpu_inference=True,
        )
        logger.info(f"GMM sampler initialized: mode_conditional={use_mode_conditional}, cls_enabled={cls_enabled}")

    # Initialize VAE prior sampler if enabled
    if vae_prior_cfg.get('enabled', False) and use_gmm:
        raise ValueError("gmm.enabled and vae_prior.enabled are mutually exclusive.")
    vae_sampler = VAEPriorSampler.from_config(vae_prior_cfg, latent_size, device, verbose=(rank == 0))
    if vae_sampler is not None and rank == 0:
        logger.info(f"VAE prior marginal diagnostics: {vae_sampler.diagnose()}")

    # Check model conditioning
    use_y_conditioning = is_model_conditional(model)
    logger.info(f"Model conditioning: use_y_conditioning={use_y_conditioning}")

    # Validate configuration
    if use_mode_conditional and not use_y_conditioning:
        logger.warning("mode_conditional=true but model has use_y_conditioning=false. "
                      "This is unusual - modes will be used for GMM sampling but model ignores them.")

    # Initialize adaptive weighter if enabled
    weighter = None
    weighting_cfg = misc.get('adaptive_weighting', {})
    if weighting_cfg.get('enabled', False):
        weighter_num_classes = gmm_sampler.num_classes if gmm_sampler is not None else num_classes
        weighter = AdaptiveClassWeighter.from_config(weighting_cfg, weighter_num_classes)
        logger.info(f"Adaptive weighting enabled: {weighting_cfg.get('strategy', 'focal')}")

        # Load weighter state if resuming from checkpoint
        if weighter_state_path is not None and os.path.exists(weighter_state_path):
            weighter.load_state(weighter_state_path)
            if rank == 0:
                logger.info(f"Loaded weighter state from {weighter_state_path}")

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
        gmm_sampler=gmm_sampler,
        vae_sampler=vae_sampler,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    start_time = time()

    # ========================================================================
    # SAMPLE GENERATION SETUP
    # ========================================================================
    # Hierarchy:
    #   use_y_conditioning: whether model uses labels (y)
    #     - True: CONDITIONAL (mode-conditional or class-conditional)
    #     - False: UNCONDITIONAL
    #   use_mode_conditional: when conditional, which type of labels
    #     - True: MODE-CONDITIONAL (y = GMM cluster modes)
    #     - False: CLASS-CONDITIONAL (y = ImageNet classes)
    # ========================================================================

    if use_y_conditioning:
        # ====================================================================
        # CONDITIONAL: model uses labels (y)
        # ====================================================================
        if use_mode_conditional:
            # ----------------------------------------------------------------
            # MODE-CONDITIONAL: y = GMM cluster modes
            # ----------------------------------------------------------------
            ys = gmm_sampler.sample_modes_weighted(micro_batch_size, device)
            using_cfg = False  # No CFG for mode-conditional
            n = ys.size(0)

            # Sample noise from GMM (required for mode-conditional)
            zs = sample_prior_noise(n, latent_size, device, latent_dtype,
                                    gmm_sampler=gmm_sampler, gmm_labels=ys)

            # Autoguidance support
            if guidance_method == "autoguidance" and guid_model_forward is not None:
                sample_model_kwargs = dict(
                    y=ys,
                    cfg_scale=guidance_scale,
                    cfg_interval=(t_min, t_max),
                    additional_model_forward=guid_model_forward,
                )
                model_fn = ema.forward_with_autoguidance
            else:
                sample_model_kwargs = dict(y=ys)
                model_fn = ema.forward

        else:
            # ----------------------------------------------------------------
            # CLASS-CONDITIONAL: y = ImageNet class labels
            # ----------------------------------------------------------------
            ys = torch.randint(num_classes, size=(micro_batch_size,), device=device)
            using_cfg = guidance_scale > 1.0
            n = ys.size(0)

            if using_cfg:
                # CFG: duplicate noise and labels
                prior_kwargs = dict(gmm_sampler=gmm_sampler, vae_sampler=vae_sampler)
                zs_cond = sample_prior_noise(n, latent_size, device, latent_dtype, **prior_kwargs)
                zs_uncond = sample_prior_noise(n, latent_size, device, latent_dtype, **prior_kwargs)
                zs = torch.cat([zs_cond, zs_uncond], dim=0)

                y_null = torch.full((n,), null_label, device=device)
                ys = torch.cat([ys, y_null], dim=0)

                sample_model_kwargs = dict(
                    y=ys,
                    cfg_scale=guidance_scale,
                    cfg_interval=(t_min, t_max),
                )
                if guidance_method == "autoguidance":
                    if guid_model_forward is None:
                        raise RuntimeError("Guidance model forward is not initialized.")
                    sample_model_kwargs["additional_model_forward"] = guid_model_forward
                    model_fn = ema.forward_with_autoguidance
                else:
                    model_fn = ema.forward_with_cfg
            else:
                # No CFG
                zs = sample_prior_noise(n, latent_size, device, latent_dtype,
                                        gmm_sampler=gmm_sampler, vae_sampler=vae_sampler)

                if guidance_method == "autoguidance" and guid_model_forward is not None:
                    sample_model_kwargs = dict(
                        y=ys,
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                        additional_model_forward=guid_model_forward,
                    )
                    model_fn = ema.forward_with_autoguidance
                else:
                    sample_model_kwargs = dict(y=ys)
                    model_fn = ema.forward

    else:
        # ====================================================================
        # UNCONDITIONAL: model ignores labels
        # ====================================================================
        n = micro_batch_size
        using_cfg = False  # CFG requires conditional model

        # Noise source: GMM / VAE prior (if enabled) or isotropic Gaussian
        zs = sample_prior_noise(n, latent_size, device, latent_dtype,
                                gmm_sampler=gmm_sampler, vae_sampler=vae_sampler)

        # Autoguidance support
        if guidance_method == "autoguidance" and guid_model_forward is not None:
            sample_model_kwargs = dict(
                y=None,  # Explicitly pass None for unconditional
                cfg_scale=guidance_scale,
                cfg_interval=(t_min, t_max),
                additional_model_forward=guid_model_forward,
            )
            model_fn = ema.forward_with_autoguidance
        else:
            sample_model_kwargs = dict()
            model_fn = ema.forward

    logger.info(f"Training for {epochs} epochs (starting from epoch {start_epoch})...")
    for epoch in range(start_epoch, epochs):
        # Set epoch for deterministic shuffling
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # Log phase transition
        if initial_phase_epochs > 0 and epoch == initial_phase_epochs:
            logger.info(f"Transitioning from initial phase to main phase (ema_decay: {initial_ema_decay} -> {ema_decay})")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.to(device)

            # ================================================================
            # ENCODE: Extract latents (and CLS tokens if needed)
            # ================================================================
            with torch.no_grad():
                # Need CLS tokens for mode prediction when GMM is enabled
                # (mode-conditional, UNCOND-GMM, or GMM + CLASS-CONDITIONAL)
                need_cls_tokens = use_gmm
                if need_cls_tokens:
                    spatial_latents, cls_latents = rae.encode(x, return_cls_token=True)
                else:
                    spatial_latents = rae.encode(x)
                    cls_latents = None

            # ================================================================
            # PREPARE model_kwargs based on conditioning mode
            # ================================================================
            # Same hierarchy as sample generation setup:
            #   use_y_conditioning -> use_mode_conditional
            if use_y_conditioning:
                # CONDITIONAL
                if use_mode_conditional:
                    # MODE-CONDITIONAL: predict modes from CLS tokens
                    with torch.no_grad():
                        cluster_modes, _ = gmm_sampler.predict_and_sample_mode(cls_latents)
                    model_kwargs = dict(y=cluster_modes)
                    y_for_weighting = cluster_modes
                else:
                    # CLASS-CONDITIONAL: use dataset labels
                    if use_gmm:
                        # GMM + CLASS-CONDITIONAL:
                        # - y = ImageNet class labels (for model conditioning)
                        # - y_gmm = predicted GMM modes (for noise sampling)
                        if gmm_sampler.cls_gmm_model is None:
                            raise RuntimeError(
                                "GMM + class-conditional requires gmm.cls_enabled=true for mode prediction."
                            )
                        with torch.no_grad():
                            cluster_modes, mode_probs = gmm_sampler.predict_and_sample_mode(cls_latents)

                        model_kwargs = dict(y=y, y_gmm=cluster_modes)
                        y_for_weighting = y  # Weight by class, not mode
                    else:
                        # Pure CLASS-CONDITIONAL: no GMM
                        model_kwargs = dict(y=y)
                        y_for_weighting = y
            else:
                # UNCONDITIONAL
                if use_gmm:
                    # UNCOND-GMM: predict modes for Transport noise sampling
                    # Model ignores these labels (use_y_conditioning=False)
                    if gmm_sampler.cls_gmm_model is None:
                        raise RuntimeError(
                            "Unconditional GMM requires gmm.cls_enabled=true for mode prediction.\n"
                            "Either enable gmm.cls_enabled or disable GMM entirely."
                        )
                    with torch.no_grad():
                        cluster_modes, _ = gmm_sampler.predict_and_sample_mode(cls_latents)
                    model_kwargs = dict(y=cluster_modes)  # Transport uses for GMM, model ignores
                    y_for_weighting = cluster_modes
                else:
                    # Pure unconditional: no labels, no GMM
                    model_kwargs = dict()
                    y_for_weighting = None

            with autocast(**autocast_kwargs):
                # Use reduction='none' if adaptive weighting is enabled
                reduction = 'none' if weighter is not None else 'mean'
                loss_dict = transport.training_losses(model, spatial_latents, model_kwargs, reduction=reduction)

                if weighter is not None and y_for_weighting is not None:
                    # Per-sample loss for adaptive weighting
                    per_sample_loss = loss_dict['loss']  # (B,)
                    weights = weighter.get_weights(y_for_weighting)
                    loss_tensor = (per_sample_loss * weights).mean()
                    weighter.update(y_for_weighting, per_sample_loss.detach())
                else:
                    # Always reduce to scalar (mean_flat returns per-sample losses)
                    loss_tensor = loss_dict['loss'].mean()

            step_loss_accum += loss_tensor.item()

            (loss_tensor / grad_accum_steps).backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            # Compute gradient norm (for logging) and clip if needed
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad if clip_grad > 0 else float('inf')
            )
            opt.step()
            schedl.step()
            # Use initial_ema_decay during initial phase, then switch to ema_decay
            current_ema_decay = initial_ema_decay if epoch < initial_phase_epochs else ema_decay
            update_ema(ema, model.module, decay=current_ema_decay)
            opt.zero_grad()

            # Load deferred optimizer state after first step (see DEFERRED OPTIMIZER
            # STATE LOADING comment above for full explanation)
            if deferred_opt_state is not None:
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    logger.info(f"Loading deferred optimizer state (GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB)")
                opt.load_state_dict(deferred_opt_state)
                del deferred_opt_state
                deferred_opt_state = None
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    logger.info(f"Deferred optimizer state loaded (GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB)")

            running_loss += step_loss_accum / grad_accum_steps
            running_grad_norm += grad_norm.item()
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                avg_grad_norm = running_grad_norm / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    log_dict = {"train loss": avg_loss, "train steps/sec": steps_per_sec, "grad_norm": avg_grad_norm}
                    wandb_utils.log(log_dict, step=train_steps)

                    # Log weighter statistics (per-mode/per-class tracking)
                    if weighter is not None and train_steps > weighter.warmup_steps:
                        stats = weighter.get_statistics()

                        # Basic statistics - logged every log_every steps
                        weighter_log = {
                            'weighter/mean_class_loss': stats['mean_loss'],
                            'weighter/std_class_loss': stats['std_loss'],
                            'weighter/min_class_loss': stats['min_loss'],
                            'weighter/max_class_loss': stats['max_loss'],
                            'weighter/loss_range': stats['max_loss'] - stats['min_loss'],
                            'weighter/loss_cv': stats['std_loss'] / (stats['mean_loss'] + 1e-8),
                        }

                        # Top 5 hardest and easiest classes
                        for i, class_id in enumerate(stats['hardest_classes'][:5]):
                            weighter_log[f'weighter/hardest_class_{i+1}_id'] = class_id
                            weighter_log[f'weighter/hardest_class_{i+1}_loss'] = stats['class_loss_ema'][class_id]

                        for i, class_id in enumerate(stats['easiest_classes'][:5]):
                            weighter_log[f'weighter/easiest_class_{i+1}_id'] = class_id
                            weighter_log[f'weighter/easiest_class_{i+1}_loss'] = stats['class_loss_ema'][class_id]

                        wandb_utils.log(weighter_log, step=train_steps)

                        # Every 1000 steps: log histograms and detailed info
                        if train_steps % 1000 == 0 and rank == 0:
                            logger.info(f"  Hardest classes: {stats['hardest_classes'][:5]}")
                            logger.info(f"  Easiest classes: {stats['easiest_classes'][:5]}")
                            logger.info(f"  Loss CV (std/mean): {stats['std_loss'] / (stats['mean_loss'] + 1e-8):.4f}")

                            # Log histogram of class losses to see distribution
                            import wandb
                            wandb_utils.log({
                                'weighter/class_loss_histogram': wandb.Histogram(stats['class_loss_ema']),
                            }, step=train_steps)

                    # Log warmup progress
                    elif weighter is not None and train_steps <= weighter.warmup_steps:
                        if train_steps % 1000 == 0 and rank == 0:
                            warmup_progress = train_steps / weighter.warmup_steps
                            logger.info(f"  Weighter warmup: {train_steps}/{weighter.warmup_steps} ({warmup_progress*100:.1f}%)")
                            wandb_utils.log({
                                'weighter/warmup_progress': warmup_progress,
                            }, step=train_steps)

                running_loss = 0.0
                running_grad_norm = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }

                    # Save weighter state if enabled
                    if weighter is not None:
                        weighter_path = f"{checkpoint_dir}/weighter_{train_steps:07d}.pt"
                        weighter.save_state(weighter_path)
                        checkpoint['weighter_path'] = weighter_path

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % sample_every == 0 or train_steps == 1:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_zs = zs

                    with autocast(**autocast_kwargs):
                        ema_result = eval_sampler(sample_zs, model_fn, **sample_model_kwargs)
                        samples = ema_result[-1]
                        del ema_result  # Free trajectory memory
                    dist.barrier()

                    if using_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    samples = rae.decode(samples.to(torch.float32))

                    # Resize if decoder output size doesn't match target image size
                    # (e.g., Scale-RAE decoder outputs 224x224 but we want 256x256)
                    if hasattr(rae, 'decoder_output_size') and rae.decoder_output_size != args.image_size:
                        samples = torch.nn.functional.interpolate(
                            samples,
                            size=(args.image_size, args.image_size),
                            mode='bicubic',
                            align_corners=False,
                        )

                    out_samples = torch.zeros(
                        (global_batch_size // grad_accum_steps, 3, args.image_size, args.image_size),
                        device=device,
                    )
                    dist.all_gather_into_tensor(out_samples, samples)
                    del samples  # Free local samples
                    if args.wandb:
                        wandb_utils.log_image(out_samples, train_steps)
                    del out_samples  # Free gathered samples
                    gc.collect()
                    torch.cuda.empty_cache()
                logger.info("Generating EMA samples done.")

            # FID evaluation (distributed across all GPUs)
            if fid_enabled and train_steps % fid_every == 0 and train_steps > 0:
                # Free GPU memory before FID evaluation
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    logger.info("Starting FID evaluation...")

                evaluate_fid(
                    rank=rank,
                    train_steps=train_steps,
                    rae=rae,
                    ema_model=ema,
                    eval_sampler=eval_sampler,
                    model_fn=model_fn,
                    sample_model_kwargs=sample_model_kwargs,
                    using_cfg=using_cfg,
                    autocast_kwargs=autocast_kwargs,
                    image_size=args.image_size,
                    num_fid_samples=fid_num_samples,
                    fid_batch_size=fid_batch_size,
                    ref_batch_path=fid_ref_batch,
                    logger=logger,
                    enable_wandb=args.wandb,
                    latent_size=latent_size,
                    gmm_sampler=gmm_sampler,
                    vae_sampler=vae_sampler,
                    num_classes=num_classes,
                    null_label=null_label,
                    world_size=world_size,
                    use_y_conditioning=use_y_conditioning,
                    use_mode_conditional=use_mode_conditional,
                    wandb_utils=wandb_utils,
                )

                # Free GPU memory after FID evaluation
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    logger.info("FID evaluation complete, resuming training...")

        if accum_counter != 0:
            if rank == 0:
                logger.warning(f"Epoch ended with incomplete gradient accumulation ({accum_counter}/{grad_accum_steps}). Resetting counter.")
            accum_counter = 0

    model.eval()
    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the train.zip archive (see ImageNetDataset).")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    main(args)
