import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import deas_critic
from openpi.models import hlg
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        self.deas = config.deas
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        if self.deas:
            self.deas_config = self.config.deas_config
            self.prefix_proj_deas = nnx.Linear(paligemma_config.width, self.deas_config.feature_dim, rngs=rngs)
            self.value_deas = deas_critic.Value(
                input_dim=self.deas_config.feature_dim + config.action_dim,
                hidden_dims=self.deas_config.hidden_dim,
                depth=4,
                output_dim=self.deas_config.num_atoms,
                rngs=rngs,
            )
            self.critic_deas = deas_critic.DoubleCritic(
                input_dim=self.deas_config.feature_dim + (self.action_horizon + 1) * config.action_dim,
                hidden_dims=[self.deas_config.hidden_dim] * 4,
                output_dim=self.deas_config.num_atoms,
                rngs=rngs,
            )
            # Initialize target critic with same structure, then copy parameters from main critic
            self.target_critic_deas = deas_critic.DoubleCritic(
                input_dim=self.deas_config.feature_dim + (self.action_horizon + 1) * config.action_dim,
                hidden_dims=[self.deas_config.hidden_dim] * 4,
                output_dim=self.deas_config.num_atoms,
                rngs=rngs,
            )
            # Copy parameters from main critic to target critic
            main_critic_state = nnx.state(self.critic_deas)
            nnx.update(self.target_critic_deas, main_critic_state)

            # compute v_min and v_max according to the discount factor
            if self.deas_config.negative_reward:
                v_min = -1 * (1 / (1 - self.deas_config.discount2))
                v_max = 0.0
            else:
                v_min = 0.0
                v_max = 1.0

            self.hlg = hlg.HLGaussLoss(
                min_value=v_min,
                max_value=v_max,
                num_bins=self.deas_config.num_atoms,
                sigma=self.deas_config.sigma,
            )

        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        if self.deas:
            return self._compute_deas_prefix(obs)

        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_deas_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b emb"], at.Float[at.Array, "b emb"]]:
        tokens, next_tokens = [], []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)

        for name in obs.next_image:
            next_image_tokens, _ = self.PaliGemma.img(obs.next_image[name], train=False)
            next_tokens.append(next_image_tokens)

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            next_tokens.append(tokenized_inputs)
        tokens = jnp.concatenate(tokens, axis=1).mean(axis=-2)
        next_tokens = jnp.concatenate(next_tokens, axis=1).mean(axis=-2)

        concat = jnp.stack([tokens, next_tokens], axis=1)  # shape: (b, 2, emb)
        concat = self.prefix_proj_deas(concat)
        concat = nnx.tanh(concat)
        tokens, next_tokens = concat[:, 0], concat[:, 1]
        return tokens, next_tokens

    def update_target_critic(self) -> None:
        """Update target critic using Polyak averaging.

        This method performs a soft update of the target critic parameters using
        the formula: target = tau * main + (1 - tau) * target
        where tau is the Polyak coefficient from the DEAS config.
        """
        if not self.deas:
            return

        tau = self.deas_config.tau
        main_state = nnx.state(self.critic_deas)
        target_state = nnx.state(self.target_critic_deas)

        # Apply Polyak update: target = tau * main + (1 - tau) * target
        updated_target_state = jax.tree.map(
            lambda main, target: tau * main + (1 - tau) * target,
            main_state,
            target_state,
        )

        # Update the target critic with the new state
        nnx.update(self.target_critic_deas, updated_target_state)

    def _value_deas_loss(
        self,
        prefix_tokens: at.Float[at.Array, "b emb"],
        observation: _model.Observation,
        actions: at.Float[at.Array, "b ah ad"],
    ) -> at.Float[at.Array, "*b"]:
        state = observation.state
        v_logits = self.value_deas(prefix_tokens, state)
        v_probs = jax.nn.softmax(v_logits, axis=-1)
        vs = self.hlg.transform_from_probs(v_probs)

        # Target critic returns (q1_logits, q2_logits), each shape (b, num_bins)
        q1_logits, q2_logits = self.target_critic_deas(
            prefix_tokens,  # VL embedding (b, emb)
            state,  # (b, ad)
            actions,  # (b, ah*ad)
        )  # -> (b, num_bins), (b, num_bins)

        # Stop gradient flow for target critic outputs and downstream computations
        q1_logits = jax.lax.stop_gradient(q1_logits)
        q2_logits = jax.lax.stop_gradient(q2_logits)

        q_logits = jnp.stack([q1_logits, q2_logits], axis=0)  # (2, b, num_bins)
        q_probs = jax.nn.softmax(q_logits, axis=-1)  # (2, b, num_bins)
        qs = jax.lax.stop_gradient(self.hlg.transform_from_probs(q_probs))  # (2, b)

        # Q aggregation logic
        if self.deas_config.q_agg == "min":
            min_q_idx = jnp.argmin(qs, axis=0)  # (b,)
            batch_indices = jnp.arange(actions.shape[0])
            # Select (b, num_bins) for each batch item from min_q_idx
            q_logit = q_logits[min_q_idx, batch_indices]  # (b, num_bins)
            q_prob = jax.nn.softmax(q_logit, axis=-1)  # (b, num_bins)
            q = self.hlg.transform_from_probs(q_prob)  # (b,)
        elif self.deas_config.q_agg == "mean":
            q_logit = jnp.mean(q_logits, axis=0)  # (b, num_bins)
            q_prob = jnp.mean(q_probs, axis=0)  # (b, num_bins)
            q = self.hlg.transform_from_probs(q_prob)  # (b,)
        else:
            raise ValueError(f"Invalid q_agg: {self.deas_config.q_agg}")

        # Expectile regression term
        g_hard = jnp.where(q >= vs, self.deas_config.expectile, 1.0 - self.deas_config.expectile)  # (b,)

        # Cross-entropy loss, using explicit log_softmax and jnp
        log_probs = jax.nn.log_softmax(v_logits, axis=-1)  # (b, num_bins)
        ce_loss = -jnp.sum(q_prob * log_probs, axis=-1)  # (b,)
        return g_hard * ce_loss  # (b,)

    def _deas_critic_loss(
        self,
        prefix_tokens: at.Float[at.Array, "b emb"],
        next_prefix_tokens: at.Float[at.Array, "b emb"],
        observation: _model.Observation,
        actions: at.Float[at.Array, "b ah ad"],
    ) -> at.Float[at.Array, "*b"]:
        state = observation.state
        done = observation.done
        reward = observation.reward
        if self.deas_config.negative_reward:
            reward -= 1

        discounts1 = self.deas_config.discount1 ** jnp.arange(self.deas_config.critic_action_horizon)
        scaled_rewards = jnp.sum(reward * discounts1, axis=-1)
        done = jnp.prod(done, axis=-1)

        # Stop gradients on everything inside this block
        v_logits = jax.lax.stop_gradient(self.value_deas(next_prefix_tokens, observation.next_state))
        v_probs = jax.nn.softmax(v_logits, axis=-1)
        vs = self.hlg.transform_from_probs(v_probs)

        target_v = (
            scaled_rewards
            + (self.deas_config.discount2 ** (self.deas_config.nstep * self.deas_config.critic_action_horizon))
            * (1.0 - done)
            * vs
        )
        target_v = jax.lax.stop_gradient(target_v)

        q1_logits, q2_logits = self.critic_deas(
            prefix_tokens, state, actions[:, : self.deas_config.critic_action_horizon]
        )
        return (self.hlg(q1_logits, target_v) + self.hlg(q2_logits, target_v)) / 2

    def _compute_deas_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, time_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        prefix_tokens, next_prefix_tokens = self.embed_deas_prefix(observation)

        value_loss = self._value_deas_loss(prefix_tokens, observation, actions)
        critic_loss = self._deas_critic_loss(prefix_tokens, next_prefix_tokens, observation, actions)

        return value_loss + critic_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"] | at.Float[at.Array, "*b"]:
        preprocess_rng, noise_rng, time_rng, deas_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        if self.deas:
            return self._compute_deas_loss(deas_rng, observation, actions, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
