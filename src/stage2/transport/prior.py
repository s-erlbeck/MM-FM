"""Source distributions for the stage-2 flow.

The flow transports x0 -> x1, where x1 is an RAE latent. x0 comes from one of three
priors: an isotropic Gaussian (default), a fitted GMM (see gmm_sampler.py), or the
PatchTokenVAE fitted by src/scripts/fit_vae_imagenet.py (this module).
"""

from typing import Optional, Tuple

import torch

from stage2.models.vae import PatchTokenVAE


class VAEPriorSampler:
    """VAE prior with posterior coupling: p(x0 | z) = N(decode(z), decoder_std^2 I).

    Training draws z ~ q(z | x1), so x0 is coupled to the very latent it is transported
    to. Inference has no x1, so it draws z ~ N(0, I). The two marginals agree only
    insofar as the aggregate posterior matches N(0, I) -- which is what the VAE's KL
    term buys. `diagnose()` reports the marginal side of that gap.

    The VAE must have been fitted with --normalize, so that its I/O space is the same
    position-wise normalized latent space the flow lives in.
    """

    def __init__(self, vae: PatchTokenVAE, latent_size: Tuple[int, ...],
                 z_shape: Tuple[int, ...], decoder_std: float = 0.0, coupled: bool = True):
        self.vae = vae
        self.latent_size = tuple(latent_size)
        self.z_shape = tuple(z_shape)
        self.decoder_std = float(decoder_std)
        self.coupled = bool(coupled)
        self.device = next(vae.parameters()).device

    @classmethod
    def from_config(cls, cfg: dict, latent_size, device, verbose: bool = True
                    ) -> Optional["VAEPriorSampler"]:
        """Build the sampler from a `vae_prior` config section. Returns None when disabled."""
        if not cfg or not cfg.get("enabled", False):
            return None

        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            raise ValueError("vae_prior.ckpt_path is required when vae_prior.enabled=true")

        vae, vae_args = load_patch_token_vae(
            ckpt_path, device, require_normalized=cfg.get("require_normalized", True)
        )

        latent_size = tuple(int(dim) for dim in latent_size)
        if vae.in_channels != latent_size[0]:
            raise ValueError(
                f"VAE in_channels ({vae.in_channels}) does not match the flow's latent "
                f"channels ({latent_size[0]}). Was the VAE fitted for this encoder?"
            )

        # Dummy round trip: fixes z_shape and turns a silent broadcast bug into a
        # startup error for the price of one forward.
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            z = vae.encode(torch.zeros(1, *latent_size, device=device)).mean
            out = vae.decode(z)
        if tuple(out.shape[1:]) != latent_size:
            raise ValueError(
                f"VAE round trip changes shape: {latent_size} -> {tuple(out.shape[1:])}. "
                "The VAE and the configured misc.latent_size disagree."
            )
        z_shape = tuple(z.shape[1:])

        sampler = cls(
            vae,
            latent_size=latent_size,
            z_shape=z_shape,
            decoder_std=cfg.get("decoder_std", 0.0),
            coupled=cfg.get("coupled", True),
        )
        if verbose:
            print(f"VAE prior loaded from {ckpt_path}: {latent_size} <-> {z_shape}, "
                  f"decoder_std={sampler.decoder_std}, coupled={sampler.coupled}, "
                  f"latent_channels={vae_args.get('latent_channels')}")
        return sampler

    def _decode_to_x0(self, z: torch.Tensor, dtype: torch.dtype,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """x0 ~ N(decode(z), decoder_std^2 I), in fp32, cast to `dtype` at the end.

        Transport.sample runs inside autocast(bf16) while x1 is fp32; bf16 GroupNorm
        would quantize the prior to ~3 decimal digits for no reason. The VAE is ~20M
        params, well under 1% of DiT-XL step FLOPs.
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            x0 = self.vae.decode(z.float())
        if self.decoder_std > 0:
            # randn_like takes no generator, so spell the shape out.
            x0 = x0 + self.decoder_std * torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )
        return x0.to(dtype)

    @torch.no_grad()
    def sample_coupled(self, x1: torch.Tensor) -> torch.Tensor:
        """Training path: z ~ q(z | x1), so x0 is coupled to this very x1."""
        if tuple(x1.shape[1:]) != self.latent_size:
            raise ValueError(f"Expected x1 of shape (B, {self.latent_size}), got {tuple(x1.shape)}")
        with torch.autocast(device_type=x1.device.type, enabled=False):
            z = self.vae.encode(x1.float()).sample()
        return self._decode_to_x0(z, x1.dtype)

    @torch.no_grad()
    def sample(self, batch_size: int, device=None, dtype: torch.dtype = torch.float32,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Inference path: z ~ N(0, I). No x1 exists, so there is nothing to couple to."""
        device = self.device if device is None else device
        z = torch.randn(batch_size, *self.z_shape, device=device, dtype=torch.float32,
                        generator=generator)
        return self._decode_to_x0(z, dtype, generator=generator)

    @torch.no_grad()
    def diagnose(self, num_samples: int = 256, seed: int = 12345) -> dict:
        """Summarize the inference-path marginal, from a dedicated fixed-seed generator.

        Using an own generator means every rank reports identical numbers with no
        collective, and the training RNG stream is left untouched. The flow's latents are
        position-wise variance-normalized, so `across_std` should land near 1.0.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x0 = self.sample(num_samples, device=self.device, generator=generator)
        stats = {
            "across_std": x0.std(dim=0).mean().item(),  # per-position variability between draws
            "global_mean": x0.mean().item(),
            "global_std": x0.std().item(),
            "min": x0.min().item(),
            "max": x0.max().item(),
        }
        if stats["across_std"] < 0.05:
            print(f"WARNING: VAE prior is near-degenerate (across_std={stats['across_std']:.4f}). "
                  "A deterministic ODE cannot transport a point mass to a full distribution. "
                  "Refit the VAE with a larger --kl-weight and/or raise vae_prior.decoder_std.")
        return stats


def load_patch_token_vae(ckpt_path: str, device, require_normalized: bool = True
                         ) -> Tuple[PatchTokenVAE, dict]:
    """Load a frozen PatchTokenVAE from a fit_vae_imagenet.py checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    vae_args = dict(ckpt.get("args", {}))

    if require_normalized and not vae_args.get("normalize", False):
        raise ValueError(
            f"{ckpt_path} was fitted without --normalize. Its I/O space is raw patch tokens, "
            "which differ from the flow's position-wise normalized latents by a per-position "
            "sqrt(var) factor that no loss curve reveals. Refit with --normalize, or set "
            "vae_prior.require_normalized=false if you know what you are doing."
        )
    if "num_groups" not in vae_args:
        # GroupNorm parameters are (C,) for any group count, so a wrong value loads
        # cleanly under strict=True and silently computes the wrong normalization.
        raise ValueError(f"{ckpt_path} has no args['num_groups']; cannot reconstruct the VAE safely.")

    vae = PatchTokenVAE(
        # the one shape parameter args does not record
        in_channels=state["encoder.conv_in.weight"].shape[1],
        latent_channels=vae_args["latent_channels"],
        hidden_channels=vae_args["hidden_channels"],
        num_res_blocks=vae_args["num_res_blocks"],
        num_groups=vae_args["num_groups"],
    )
    vae.load_state_dict(state, strict=True)
    return vae.to(device).eval().requires_grad_(False), vae_args


def sample_prior_noise(batch_size: int, latent_size, device, dtype, *,
                       gmm_sampler=None, vae_sampler=None, gmm_labels=None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Draw the flow's source noise from whichever prior is configured.

    For the *inference* paths only, where no x1 exists to couple to -- the training path
    lives in Transport.sample. Returns just the noise: GMMSampler's per-sample variance is
    discarded at every call site. With gmm_sampler set and gmm_labels None, modes come
    from the component weights.
    """
    if gmm_sampler is not None and vae_sampler is not None:
        raise ValueError("The flow has one source distribution: pass gmm_sampler or vae_sampler.")

    if gmm_sampler is not None:
        if gmm_labels is None:
            gmm_labels = gmm_sampler.sample_modes_weighted(batch_size, device, generator=generator)
        noise, _ = gmm_sampler.sample(gmm_labels, latent_size, device, dtype)
        return noise
    if vae_sampler is not None:
        return vae_sampler.sample(batch_size, device, dtype, generator=generator)
    return torch.randn(batch_size, *latent_size, device=device, dtype=dtype, generator=generator)
