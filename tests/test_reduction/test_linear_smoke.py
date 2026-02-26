import numpy as np

from boed.reduction.linear import KLE, PCA, POD


def test_pca_fit_transform_inverse_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12))

    pca = PCA(n_components=5)
    Xr = pca.fit_transform(X)
    Xhat = pca.inverse_transform(Xr)

    assert Xr.shape == (40, 5)
    assert Xhat.shape == X.shape
    assert np.all(np.isfinite(Xr))


def test_kle_fit_with_n_modes():
    def cov_func(z, zp):
        z = np.atleast_1d(z)
        zp = np.atleast_1d(zp)
        d = np.abs(z[:, None] - zp[None, :])
        return np.exp(-(d ** 2) / 0.1)

    kle = KLE(domain=(0.0, 1.0), n_points=25)
    kle.fit(cov_func, n_modes=6)

    assert kle.eigenvalues.shape == (6,)
    assert kle.eigenfunctions.shape == (25, 6)
    assert kle.n_components == 6


def test_pod_fit_and_reconstruct():
    rng = np.random.default_rng(1)
    snapshots = rng.normal(size=(30, 20))
    pod = POD(energy_threshold=0.95)
    pod.fit(snapshots, use_snapshot_method=True)

    coords = pod.transform(snapshots)
    recon = pod.inverse_transform(coords)

    assert recon.shape == snapshots.shape
    assert coords.shape[0] == pod.n_components
