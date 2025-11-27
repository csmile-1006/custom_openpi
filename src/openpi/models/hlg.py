import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class HLGaussLoss(nnx.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma

    @at.typecheck
    def __call__(self, logits: at.Float[at.Array, "*b num_bins"], target: at.Float[at.Array, "*b"]) -> at.Float[at.Array, "*b"]:
        target_probs = self.transform_to_probs(target)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(target_probs * log_probs, axis=-1)

    @at.typecheck
    def transform_to_probs(self, target: at.Float[at.Array, "*b"]) -> at.Float[at.Array, "*b num_bins"]:
        support = jnp.linspace(self.min_value, self.max_value, self.num_bins + 1)
        cdf_evals = jax.scipy.special.erf(
            (support - target[..., None]) / (jnp.sqrt(2.0) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z[..., None]

    @at.typecheck
    def transform_from_probs(self, probs: at.Float[at.Array, "*b num_bins"]) -> at.Float[at.Array, "*b"]:
        support = jnp.linspace(self.min_value, self.max_value, self.num_bins + 1)
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers, axis=-1)
