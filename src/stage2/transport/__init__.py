from .transport import Transport, ModelType, WeightType, PathType, Sampler
from .prior import VAEPriorSampler, sample_prior_noise

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
    gmm_sampler=None,
    vae_sampler=None,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - prediction: model prediction type (velocity, noise, score)
    - loss_weight: weight loss by velocity or likelihood
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    - time_dist_type: type of time distribution to use; default to uniform
    - time_dist_shift: shift for time distribution; default to 1.0
    - gmm_sampler: GMMSampler object for noise sampling (optional)
    - vae_sampler: VAEPriorSampler object for noise sampling (optional, exclusive with gmm_sampler)
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
        gmm_sampler=gmm_sampler,
        vae_sampler=vae_sampler,
    )

    return state