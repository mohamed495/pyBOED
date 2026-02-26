import numpy as np


def test_legacy_inference_imports_still_resolve():
    from boed.core.posterior import LinearGaussianModel as LegacyLinearGaussianModel
    from boed.core.posterior import NonLinearInverseLaplace
    from boed.inference import LinearGaussianModel, NonlinearLaplaceModel

    assert LegacyLinearGaussianModel is LinearGaussianModel
    assert NonlinearLaplaceModel is NonLinearInverseLaplace


def test_legacy_priors_kernels_gp_prior_reexport_matches_canonical():
    from boed.priors.gp_priors import GaussianProcessPrior as CanonicalGP
    from boed.priors.kernels import GaussianProcessPrior as LegacyGP

    assert LegacyGP is CanonicalGP


def test_legacy_observation_matrix_alias_matches_observation_operator():
    from boed.observations import SpaceTimeSensors

    sensors = SpaceTimeSensors(x_idx=[0, 2], t_idx=[0, 1], nx=4)
    H_old = sensors.observation_matrix(nt=3)
    H_new = sensors.observation_operator(nt=3)

    assert H_old.shape == (2, 12)
    assert np.allclose(H_old, H_new)
