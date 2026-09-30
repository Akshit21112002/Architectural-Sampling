"""
Recurrent Visual Depth (RVD) for Qwen2.5-VL — v2 (layer-level patching).

Why v2: in recent transformers refactors, Qwen2_5_VLModel.forward iterates
over language_model.layers directly and may never dispatch through
language_model.forward(). Patching .forward() therefore silently no-ops
(this is the bug that produced L2(K=1, K=8) = 0.0 in v1).

v2 hooks into the LAYERS themselves, which are always called regardless
of which parent owns the layer loop:

  - Layer `block_start` is replaced with a RecurrentBlockHead that runs
    the FULL block (layers [block_start..block_end]) K times internally.
  - Layers `block_start+1 .. block_end` become IdentityLayer (the block
    head already ran them).
  - K=1 reduces exactly to one block pass through the original layers.

This patch survives transformers refactors because layer.forward is the
stable interface — parents always call layer(hidden_states, ...) in order.

Public API:
    patch_model(model, block_start, block_end, tokenizer=None) -> model
    set_K(model, K)
    get_K(model) -> int
    set_mode(model, mode)            # select the recurrent-evolution variant
    get_mode(model) -> str
    unpatch_model(model) -> model

The evolution variant is chosen with the EVOLUTION_MODE constant below (or
overridden per-model at runtime with set_mode).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"


# =============================================================================
# EVOLUTION MODE  (select the recurrent-refinement variant here)
# =============================================================================
# Controls WHICH token positions the recurrent block iterates on, and HOW the
# refined states are read back out. Set this once at the top of the script, or
# override per-model at runtime with set_mode(model, "...").
#
#   "vision"          (default) Iterate ONLY vision tokens; text is held frozen
#                     at its canonical (first-pass) state. Read back the final
#                     (damped) iterate. This is the ORIGINAL RVD behavior and is
#                     exactly equivalent to the pre-refactor code.
#
#   "all"             Iterate EVERY token (vision + text): whole-sequence
#                     recurrent depth. Fires during multi-token prefill only.
#
#   "language"        Iterate ONLY text/language tokens; vision held frozen at
#                     canonical. The mirror image of "vision" — a control to
#                     test whether refinement helps specifically via the vision
#                     tokens or just from the extra depth applied anywhere.
#
#   "vision_average"  Iterate vision tokens exactly like "vision", but read back
#                     the MEAN over all K block passes instead of only the last
#                     iterate. A smoother averaging-refinement baseline.
#
#   "none"            No recurrence; always return the canonical pass. Exactly
#                     equivalent to K=1. Pure baseline / A-B control.
#
EVOLUTION_MODE = "all"

_VALID_MODES = ("vision", "all", "language", "vision_average", "none")


# -----------------------------------------------------------------------------
# Mask construction
# -----------------------------------------------------------------------------

def _build_vision_mask(
    input_ids: torch.LongTensor,
    image_pad_id: int,
    video_pad_id: Optional[int] = None,
) -> torch.BoolTensor:
    mask = input_ids.eq(image_pad_id)
    if video_pad_id is not None and video_pad_id >= 0:
        mask = mask | input_ids.eq(video_pad_id)
    return mask


# -----------------------------------------------------------------------------
# RecurrentBlockHead — replaces layer[block_start]
# -----------------------------------------------------------------------------

class RecurrentBlockHead(nn.Module):
    """
    Wraps the full block [block_start..block_end]. On forward:
      1. Reads K, the evolution mode, and the current vision mask from the
         parent model.
      2. Runs the full block K times, keeping the "frozen" positions (whatever
         the mode does not target) at the canonical (first-pass) output and
         iterating only the "evolve" positions.
      3. Returns the final hidden state.
    Downstream IdentityLayer instances then pass this through unchanged,
    so the parent's layer loop sees the right number of "layer outputs".
    """

    def __init__(self, block_layers: nn.ModuleList, model_ref_holder: dict):
        super().__init__()
        self.block_layers = block_layers
        self._model_ref_holder = model_ref_holder
        # Qwen2.5-VL's parent forward reads `decoder_layer.attention_type` on
        # every layer to pick the right causal mask. We mirror it from the
        # first layer of the block, since that's the layer we're "standing in
        # for" from the parent's point of view.
        first = block_layers[0]
        if hasattr(first, "attention_type"):
            self.attention_type = first.attention_type
        # Some transformers versions also read these — forward them defensively.
        for attr in ("layer_idx", "self_attn"):
            if hasattr(first, attr):
                try:
                    setattr(self, attr, getattr(first, attr))
                except Exception:
                    pass

        # ---- Stability scalars for recurrent iteration ----
        # Plain iteration (x_iter = block(x_iter)) of a transformer block
        # diverges geometrically because the block has internal residual
        # adds; iterating it grows the residual stream norm. We stabilize
        # the iteration with two scalars, exactly analogous to Geiping et
        # al.'s recurrent block (Huginn):
        #
        #   inp = (1 - alpha) * x_iter + alpha * canonical    # re-inject anchor
        #   new = block(inp)
        #   x_iter = (1 - beta) * x_iter + beta * new          # damped update
        #
        # With (alpha=0, beta=1) you recover the original free iteration
        # (which blows up). (alpha=0.3, beta=0.5) is a safe default for
        # frozen weights: it keeps the residual norm bounded while still
        # letting the block refine vision tokens.
        #
        # These can be made learnable parameters later — for now they're
        # fixed scalars so the smoke test produces stable, untrained behavior.
        self.alpha_anchor = 0   # re-injection strength toward canonical
        self.beta_update = 1    # how much of the new block output to absorb

    def _read_state(self):
        ref = self._model_ref_holder.get("model", None)
        if ref is None:
            return 1, None, "none"
        K = getattr(ref, "_rvd_K", 1)
        vmask = getattr(ref, "_rvd_current_vmask", None)
        mode = getattr(ref, "_rvd_mode", EVOLUTION_MODE)
        return K, vmask, mode

    def _forward_block(self, x, *args, **kwargs):
        """One full pass through the wrapped block layers."""
        for layer in self.block_layers:
            out = layer(x, *args, **kwargs)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def _build_evolve_mask(self, mode, vmask, hidden_states):
        """
        Return a [B, T] boolean tensor marking positions to iterate, or None
        if the iteration should be skipped entirely (fall back to canonical).

        Guards (shared across modes):
          - Never iterate on a single-token step (autoregressive decode):
            running the block K times there would write to the KV cache K
            times and corrupt generation. Recurrence fires on multi-token
            prefill / scoring only.
          - All modes rely on the captured vision mask to (a) confirm we are in
            a real, cache-aligned prefill and (b) locate the vision tokens.
        """
        seq_len = hidden_states.shape[1]
        if seq_len <= 1:
            return None
        if vmask is None or vmask.shape[1] != seq_len:
            return None

        if mode == "all":
            return torch.ones_like(vmask, dtype=torch.bool)
        if mode in ("vision", "vision_average"):
            return vmask if vmask.any() else None
        if mode == "language":
            text_mask = ~vmask
            return text_mask if text_mask.any() else None
        # Unknown mode -> be safe, skip evolution.
        return None

    def forward(self, hidden_states, *args, **kwargs):
        K, vmask, mode = self._read_state()

        # ----- Canonical pass (always runs) -----
        canonical = self._forward_block(hidden_states, *args, **kwargs)

        # Baseline / disabled evolution, or trivial K: exact vanilla behavior.
        if mode == "none" or K == 1:
            return (canonical,)

        # Which positions iterate vs. stay pinned to canonical (mode-dependent).
        evolve_mask = self._build_evolve_mask(mode, vmask, hidden_states)
        if evolve_mask is None:
            # No valid mask (no input_ids, cache-misaligned/decode step, or no
            # target tokens present) -> fall back to canonical.
            return (canonical,)

        # ----- K-1 additional iterations, stabilized -----
        # device_map="auto" can shard the recurrent block across GPUs (e.g.
        # block_layers[0..3] on cuda:0, block_layers[4..6] on cuda:1).
        # After the canonical pass, `canonical` lives on the LAST layer's
        # device. We pin all combination ops to canonical's device and let
        # accelerate's hooks auto-move tensors back to the first layer's
        # device when we feed `inp` into the block.
        dev = canonical.device
        a = self.alpha_anchor
        b = self.beta_update

        # Move the evolve mask to canonical's device once (tiny boolean tensor)
        if evolve_mask.device != dev:
            evolve_mask = evolve_mask.to(dev)
        mask_b = evolve_mask.unsqueeze(-1)  # [B, T, 1]

        # Averaging read-out: mean over all K block passes (canonical + iters)
        # instead of the last iterate. Same trajectory, different aggregation.
        average = mode.endswith("_average")
        if average:
            accum = canonical      # canonical counts as pass 1
            count = 1

        x_iter = canonical
        for k in range(1, K):
            # Ensure x_iter is on dev (output of previous block pass may have
            # ended up there already, but be defensive)
            if x_iter.device != dev:
                x_iter = x_iter.to(dev)
            # Re-inject the canonical anchor at each step
            anchored = (1.0 - a) * x_iter + a * canonical
            # Evolve positions get the anchored input; frozen positions stay
            # at canonical.
            inp = torch.where(mask_b, anchored, canonical)
            # Feed back into the block — accelerate hooks will move `inp` to
            # the first layer's device automatically.
            x = self._forward_block(inp, *args, **kwargs)
            # x now lives on dev (last layer's device). Damped update.
            if x.device != dev:
                x = x.to(dev)
            x_iter = (1.0 - b) * x_iter + b * x
            if average:
                accum = accum + x
                count += 1

        evolved = (accum / count) if average else x_iter
        final = torch.where(mask_b, evolved, canonical)
        return (final,)


class IdentityLayer(nn.Module):
    """Replaces layers [block_start+1 .. block_end] after the head runs."""

    def __init__(self, ref_layer: nn.Module):
        super().__init__()
        # Hold the original layer so its weights are still in the graph for
        # state_dict, etc., but the RecurrentBlockHead is the one calling it.
        self._ref = ref_layer
        # Mirror attributes the parent's layer loop may read off each layer.
        if hasattr(ref_layer, "attention_type"):
            self.attention_type = ref_layer.attention_type
        for attr in ("layer_idx", "self_attn"):
            if hasattr(ref_layer, attr):
                try:
                    setattr(self, attr, getattr(ref_layer, attr))
                except Exception:
                    pass

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


# -----------------------------------------------------------------------------
# Pre-forward hook on top-level model: capture input_ids -> vision_mask
# -----------------------------------------------------------------------------

def _make_pre_forward_hook(model):
    def hook(mod, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        if input_ids is None:
            model._rvd_current_vmask = None
            return None
        image_pad_id = getattr(model, "_rvd_image_pad_id", None)
        video_pad_id = getattr(model, "_rvd_video_pad_id", None)
        if image_pad_id is None:
            model._rvd_current_vmask = None
            return None
        model._rvd_current_vmask = _build_vision_mask(
            input_ids, image_pad_id, video_pad_id
        )
        return None
    return hook


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def patch_model(model, block_start: int, block_end: int, tokenizer=None):
    text_model = _get_text_model(model)
    L = len(text_model.layers)
    assert 0 <= block_start <= block_end < L, (
        f"block range [{block_start}, {block_end}] out of bounds for {L} layers"
    )

    # Qwen2.5-VL uses mixed full/window attention. The parent forward picks a
    # different causal mask per layer based on layer.attention_type. Our
    # RecurrentBlockHead receives ONE mask from the parent (the one for its
    # own attention_type) and forwards it to every inner block layer — so
    # all layers in the block must share the same attention_type.
    layers = list(text_model.layers)
    types_in_block = []
    for i in range(block_start, block_end + 1):
        t = getattr(layers[i], "attention_type", None)
        types_in_block.append(t)
    if len(set(types_in_block)) > 1:
        fullatt = getattr(model.config, "fullatt_block_indexes", None) \
            or getattr(getattr(model.config, "text_config", model.config),
                       "fullatt_block_indexes", None)
        raise ValueError(
            f"Recurrent block [{block_start}, {block_end}] straddles layers "
            f"with different attention_type values: {types_in_block}. "
            f"Pick a block that does not cross a full-attention layer. "
            f"For this model, full-attention layer indices are: {fullatt}. "
            f"E.g. valid blocks for Qwen2.5-VL-3B include [0,6], [8,14], "
            f"[16,22], [24,30], or any subrange within them."
        )

    image_pad_id, video_pad_id = _resolve_pad_ids(model, tokenizer)
    model._rvd_image_pad_id = image_pad_id
    model._rvd_video_pad_id = video_pad_id if video_pad_id is not None else -1
    model._rvd_K = 1
    model._rvd_current_vmask = None
    # Evolution variant: seed from the top-of-script constant (validated).
    model._rvd_mode = EVOLUTION_MODE if EVOLUTION_MODE in _VALID_MODES else "vision"

    ref_holder = {"model": model}

    if not hasattr(text_model, "_rvd_orig_layers"):
        text_model._rvd_orig_layers = list(text_model.layers)
        text_model._rvd_block_start = block_start
        text_model._rvd_block_end = block_end

    block_layers = nn.ModuleList(
        [text_model._rvd_orig_layers[i] for i in range(block_start, block_end + 1)]
    )

    new_layers = nn.ModuleList()
    for i, layer in enumerate(text_model._rvd_orig_layers):
        if i == block_start:
            new_layers.append(RecurrentBlockHead(block_layers, ref_holder))
        elif block_start < i <= block_end:
            new_layers.append(IdentityLayer(layer))
        else:
            new_layers.append(layer)
    text_model.layers = new_layers

    if not hasattr(model, "_rvd_hook_handle"):
        h = model.register_forward_pre_hook(
            _make_pre_forward_hook(model), with_kwargs=True
        )
        model._rvd_hook_handle = h

    return model


def unpatch_model(model):
    text_model = _get_text_model(model)
    if hasattr(text_model, "_rvd_orig_layers"):
        text_model.layers = nn.ModuleList(text_model._rvd_orig_layers)
        del text_model._rvd_orig_layers
    if hasattr(model, "_rvd_hook_handle"):
        model._rvd_hook_handle.remove()
        del model._rvd_hook_handle
    for attr in ("_rvd_K", "_rvd_current_vmask", "_rvd_mode",
                 "_rvd_image_pad_id", "_rvd_video_pad_id"):
        if hasattr(model, attr):
            delattr(model, attr)
    return model


def set_K(model, K: int) -> None:
    assert K >= 1
    model._rvd_K = K


def get_K(model) -> int:
    return getattr(model, "_rvd_K", 1)


def set_mode(model, mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown evolution mode {mode!r}. Valid modes: {_VALID_MODES}"
        )
    model._rvd_mode = mode


def get_mode(model) -> str:
    return getattr(model, "_rvd_mode", EVOLUTION_MODE)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_text_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "language_model") \
            and hasattr(model.model.language_model, "layers"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError(
        "Could not locate text model. Run diagnose_patch_site.py to inspect."
    )


def _resolve_pad_ids(model, tokenizer) -> Tuple[int, Optional[int]]:
    if tokenizer is not None:
        image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        video_pad_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD_TOKEN)
        if image_pad_id == tokenizer.unk_token_id:
            image_pad_id = None
        if image_pad_id is not None:
            return image_pad_id, video_pad_id

    cfg = model.config
    image_pad_id = getattr(cfg, "image_token_id", None)
    video_pad_id = getattr(cfg, "video_token_id", None)
    if image_pad_id is None:
        raise RuntimeError("Could not resolve <|image_pad|> token id.")
    return image_pad_id, video_pad_id