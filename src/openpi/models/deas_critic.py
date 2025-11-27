import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@at.typecheck
class LayerNorm(nnx.Module):
    """Layer normalization module for NNX."""

    def __init__(self, features: int, eps: float = 1e-5, rngs: nnx.Rngs | None = None):
        self.features = features
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((features,)))
        self.bias = nnx.Param(jnp.zeros((features,)))

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b d"]:
        """Apply layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normed = (x - mean) / jnp.sqrt(variance + self.eps)
        return normed * self.scale + self.bias


@at.typecheck
class MLP(nnx.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, rngs: nnx.Rngs, *, layer_norm: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layer_norm = layer_norm

        # Build layers using nnx.Dict instead of list
        current_dim = input_dim
        self.linears = nnx.Dict()
        self.norms = nnx.Dict()
        for i, hidden_dim in enumerate(hidden_dims):
            self.linears[f"linear_{i}"] = nnx.Linear(current_dim, hidden_dim, rngs=rngs)
            if layer_norm:
                self.norms[f"norm_{i}"] = LayerNorm(hidden_dim, rngs=rngs)
            else:
                self.norms[f"norm_{i}"] = None
            current_dim = hidden_dim
        self.linears["linear_out"] = nnx.Linear(current_dim, output_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b out"]:
        """Forward pass through MLP."""
        for i in range(len(self.hidden_dims)):
            linear = self.linears[f"linear_{i}"]
            norm = self.norms[f"norm_{i}"]
            x = linear(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.gelu(x)
        # Final layer (no activation)
        return self.linears["linear_out"](x)

@at.typecheck
class ResidualBlock(nnx.Module):
    def __init__(self, hidden_dims: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(hidden_dims, hidden_dims, rngs=rngs)
        self.norm1 = LayerNorm(hidden_dims, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims, hidden_dims, rngs=rngs)
        self.norm2 = LayerNorm(hidden_dims, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dims, hidden_dims, rngs=rngs)
        self.norm3 = LayerNorm(hidden_dims, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b d"]:
        """Forward pass through residual block."""
        identity = x
        # First dense block
        x = self.linear1(x)
        x = self.norm1(x)
        x = jax.nn.relu(x)
        # Second dense block
        x = self.linear2(x)
        x = self.norm2(x)
        x = jax.nn.relu(x)
        # Final transformation before residual connection
        x = self.linear3(x)
        x = self.norm3(x)
        return x + identity


@at.typecheck
class BRONet(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.add_final_layer = add_final_layer
        self.output_dim = output_dim

        if rngs is None:
            rngs = nnx.Rngs()

        # Create the residual blocks based on depth
        self.input_projection = nnx.Linear(input_dim, hidden_dims, rngs=rngs)
        self.input_layernorm = LayerNorm(hidden_dims, rngs=rngs)
        self.residual_blocks = nnx.Dict()
        for i in range(depth):
            setattr(self, f"residual_block_{i}", ResidualBlock(hidden_dims, rngs=rngs))
            self.residual_blocks[f"block_{i}"] = getattr(self, f"residual_block_{i}")

        if add_final_layer:
            self.final_layer = nnx.Linear(hidden_dims, output_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b out"]:
        """Forward pass through BRONet."""
        x = self.input_projection(x)
        x = self.input_layernorm(x)
        x = jax.nn.relu(x)
        for block_name in self.residual_blocks:
            x = self.residual_blocks[block_name](x)

        if self.add_final_layer:
            x = self.final_layer(x)

        return x

@at.typecheck
class DoubleCritic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.Q1 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )
        self.Q2 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        vl_embed_features: at.Float[at.Array, "*b emb"],
        states: at.Float[at.Array, "*b s"],
        actions: at.Float[at.Array, "*b a"],
    ) -> tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b"]]:
        """Forward pass through double critic."""
        batch_size = states.shape[0]
        state_action = jnp.concatenate(
            [
                jnp.reshape(vl_embed_features, (batch_size, -1)),
                jnp.reshape(states, (batch_size, -1)),
                jnp.reshape(actions, (batch_size, -1)),
            ],
            axis=1,
        )
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        if q1.shape[-1] == 1:
            q1 = jnp.squeeze(q1, axis=-1)
        if q2.shape[-1] == 1:
            q2 = jnp.squeeze(q2, axis=-1)
        return q1, q2


@at.typecheck
class BRONetDoubleCritic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.Q1 = BRONet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
            rngs=rngs,
        )
        self.Q2 = BRONet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        vl_embed_features: at.Float[at.Array, "*b emb"],
        states: at.Float[at.Array, "*b s"],
        actions: at.Float[at.Array, "*b a"],
    ) -> tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b"]]:
        """Forward pass through BRONet double critic."""
        batch_size = states.shape[0]
        state_action = jnp.concatenate(
            [
                jnp.reshape(vl_embed_features, (batch_size, -1)),
                jnp.reshape(states, (batch_size, -1)),
                jnp.reshape(actions, (batch_size, -1)),
            ],
            axis=1,
        )
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)
        if q1.shape[-1] == 1:
            q1 = jnp.squeeze(q1, axis=-1)
        if q2.shape[-1] == 1:
            q2 = jnp.squeeze(q2, axis=-1)
        return q1, q2


@at.typecheck
class Value(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        vl_embed_features: at.Float[at.Array, "*b emb"],
        states: at.Float[at.Array, "*b s"],
    ) -> at.Float[at.Array, "*b"]:
        """Forward pass through value network."""
        batch_size = states.shape[0]
        v = self.value(
            jnp.concatenate(
                [jnp.reshape(vl_embed_features, (batch_size, -1)), jnp.reshape(states, (batch_size, -1))],
                axis=1,
            )
        )
        if v.shape[-1] == 1:
            v = jnp.squeeze(v, axis=-1)
        return v


@at.typecheck
class BRONetValue(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        vl_embed_features: at.Float[at.Array, "*b emb"],
        states: at.Float[at.Array, "*b s"],
    ) -> at.Float[at.Array, "*b"]:
        """Forward pass through BRONet value network."""
        batch_size = states.shape[0]
        v = self.value(
            jnp.concatenate(
                [jnp.reshape(vl_embed_features, (batch_size, -1)), jnp.reshape(states, (batch_size, -1))],
                axis=1,
            )
        )
        if v.shape[-1] == 1:
            v = jnp.squeeze(v, axis=-1)
        return v


@at.typecheck
class StateValue(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(self, states: at.Float[at.Array, "*b s"]) -> at.Float[at.Array, "*b"]:
        """Forward pass through state value network."""
        batch_size = states.shape[0]
        v = self.value(jnp.reshape(states, (batch_size, -1)))
        if v.shape[-1] == 1:
            v = jnp.squeeze(v, axis=-1)
        return v


@at.typecheck
class StateDoubleCritic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        rngs: nnx.Rngs | None = None,
        *,
        add_final_layer: bool = True,
        output_dim: int = 1,
    ):
        if rngs is None:
            rngs = nnx.Rngs()
        self.Q1 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )
        self.Q2 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        states: at.Float[at.Array, "*b s"],
        actions: at.Float[at.Array, "*b a"],
    ) -> tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b"]]:
        """Forward pass through state double critic."""
        batch_size = states.shape[0]
        state_action = jnp.concatenate(
            [jnp.reshape(states, (batch_size, -1)), jnp.reshape(actions, (batch_size, -1))],
            axis=1,
        )
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        if q1.shape[-1] == 1:
            q1 = jnp.squeeze(q1, axis=-1)
        if q2.shape[-1] == 1:
            q2 = jnp.squeeze(q2, axis=-1)
        return q1, q2
