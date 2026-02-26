import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional


class RNLA:
    """Randomized numerical linear algebra helpers.

    Provides range-finder variants and a simple randomized SVD wrapper.
    """

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A)

    def rangefinder(self, l: int) -> np.ndarray:
        W = np.random.normal(size=(self.A.shape[1], l))
        Y = self.A @ W
        Q, _ = np.linalg.qr(Y, mode="reduced")
        return Q

    def kyrlov_rangefinder(self, l: int, q: int) -> np.ndarray:
        W = np.random.normal(size=(self.A.shape[1], l))
        Y = W.copy()
        for _ in range(q):
            W = self.A @ (self.A.T @ W)
            W, _ = np.linalg.qr(W, mode="reduced")
            Y = np.hstack((Y, W))
        Q, _ = np.linalg.qr(Y, mode="reduced")
        return Q

    def randomized_svd(self, k: int, p: int, q: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if q != 0:
            Q = self.kyrlov_rangefinder(l=k + p, q=q)
        else:
            Q = self.rangefinder(l=k + p)
        B = Q.T @ self.A
        U_hat, s, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat
        return U[:, :k], s[:k], Vh[:k, :]

    def plot_singular_values_for_qs(self, k: int, p: int, qs: Sequence[int], labels=None, log_scale=False, filename=None):
        plt.figure(figsize=(8, 6))
        for i, q in enumerate(qs):
            _, s, _ = self.randomized_svd(k=k, p=p, q=q)
            idx = np.arange(1, len(s) + 1)
            label = labels[i] if (labels is not None and i < len(labels)) else f"q={q}"
            plt.plot(idx, s, marker='o', linestyle='-', label=label)

        plt.xlabel('Index i')
        plt.ylabel('Singular values $s_i$')
        if log_scale:
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()


# Convenience wrappers
def rangefinder(A: np.ndarray, l: int) -> np.ndarray:
    return RNLA(A).rangefinder(l=l)


def randomized_svd(A: np.ndarray, k: int, p: int, q: int = 0):
    return RNLA(A).randomized_svd(k=k, p=p, q=q)


def assessment(A: np.ndarray, Atilde: np.ndarray) -> float:
    return np.linalg.norm(A - Atilde, 2) / np.linalg.norm(A, 2)


def _to_float_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype.kind in "ui":
        max_val = np.iinfo(img.dtype).max
        return img.astype(np.float64) / max_val

    img = img.astype(np.float64)
    max_val = float(np.max(img)) if img.size else 1.0
    if max_val > 1.0:
        img = img / (255.0 if max_val <= 255.0 else max_val)
    return img


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    return np.tensordot(img[..., :3], weights, axes=([-1], [0]))


def low_rank_approx(A: np.ndarray, k: int, p: int, q: int = 0) -> np.ndarray:
    U, s, Vh = RNLA(A).randomized_svd(k=k, p=p, q=q)
    return (U * s) @ Vh


def demo_image(
    image_path: str,
    k: int = 50,
    p: int = 5,
    q: int = 0,
    grayscale: bool = True,
    filename: Optional[str] = "rnla_image_demo.png",
):
    """Demo RNLA on an image by low-rank approximation.

    Parameters
    ----------
    image_path : str
        Path to the input image (PNG recommended).
    k, p, q : int
        Randomized SVD parameters.
    grayscale : bool
        If True, convert image to grayscale before approximation.
    filename : str or None
        Output path for the side-by-side comparison figure. If None, uses plt.show().
    """
    img = _to_float_image(plt.imread(image_path))
    if img.ndim == 3 and img.shape[2] >= 4:
        img = img[..., :3]

    if grayscale:
        if img.ndim == 3:
            A = _rgb_to_gray(img)
        else:
            A = img
        Atilde = low_rank_approx(A, k=k, p=p, q=q)
        rel_err = np.linalg.norm(A - Atilde) / np.linalg.norm(A)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(A, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original (gray)")
        axes[1].imshow(np.clip(Atilde, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"RNLA k={k}, p={p}, q={q}\nrel err={rel_err:.3e}")
    else:
        if img.ndim == 2:
            A = img
            Atilde = low_rank_approx(A, k=k, p=p, q=q)
            rel_err = np.linalg.norm(A - Atilde) / np.linalg.norm(A)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(A, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Original (gray)")
            axes[1].imshow(np.clip(Atilde, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"RNLA k={k}, p={p}, q={q}\nrel err={rel_err:.3e}")
        else:
            A = img
            Atilde = np.zeros_like(A)
            for c in range(3):
                Atilde[..., c] = low_rank_approx(A[..., c], k=k, p=p, q=q)
            rel_err = np.linalg.norm(A - Atilde) / np.linalg.norm(A)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(A, vmin=0, vmax=1)
            axes[0].set_title("Original (RGB)")
            axes[1].imshow(np.clip(Atilde, 0, 1), vmin=0, vmax=1)
            axes[1].set_title(f"RNLA k={k}, p={p}, q={q}\nrel err={rel_err:.3e}")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def demo_plot(filename: str = 'rnla_singulars_qs.png'):
    """Quick demo: build a small matrix and save singular value plots."""
    N = 1000
    A = np.random.randn(N, N)
    k = 50
    p = 5
    qs = [0, 1, 2, 3]
    RNLA(A).plot_singular_values_for_qs(k=k, p=p, qs=qs, log_scale=True, filename=filename)


if __name__ == "__main__":
    demo_plot()
