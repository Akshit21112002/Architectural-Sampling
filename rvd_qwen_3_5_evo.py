"""
Recurrent Visual Depth (RVD) for Qwen3.5 — v5 (Qwen3.5 only, multi-mode).

Scope: this file targets Qwen3.5 specifically (Qwen3_5ForConditionalGeneration,
Qwen3_5DecoderLayer). It is the Qwen3.5 port of the Qwen3-VL-only RVD patcher;
the public API, the evolution modes, and the recurrent-refinement logic are
IDENTICAL. Only the model-specific plumbing (where the text backbone lives, how
vision-placeholder tokens are identified, and a few doc comments) changed.

Why Qwen3.5 needs its own file / what actually changed vs. Qwen3-VL
-------------------------------------------------------------------
Qwen3.5 is a natively-multimodal foundation model. Two architectural facts
matter for this patcher, both confirmed against transformers' modeling_qwen3_5:

  1. Single decoder-layer class. The text backbone (a Qwen3_5TextModel, reused
     from Qwen3-Next's linear-attention decoder) stacks ONE class,
     `Qwen3_5DecoderLayer`, for every layer. Each instance internally holds
     either a Gated DeltaNet linear-attention mixer (`self.linear_attn`) or a
     full softmax-attention mixer (`self.self_attn`), selected per index by
     `config.layer_types[i] in {"linear_attention", "full_attention"}` (a 3:1
     DeltaNet:Attention hybrid stack). Because the class is the SAME for both
     mixer kinds, `text_model.layers` is homogeneous in type, and:
       - transformers' `_can_record_outputs = {"hidden_states": Qwen3_5DecoderLayer}`
         resolves against our shims (they subclass that class → isinstance True),
         so the @capture_outputs hook installer registers naturally.
       - `type(text_model._rvd_orig_layers[0])` is always `Qwen3_5DecoderLayer`
         regardless of whether layer 0 is a linear or full layer, so the shim
         classes are built off the correct base. No class name is hardcoded.

  2. Hybrid (KV + recurrent) cache. Full-attention layers use a KV cache; the
     DeltaNet layers keep a *recurrent* state (a conv state + a linear-attention
     "delta-rule" state) instead of K,V. In the layer forward the cache is
     threaded in as `cache_params=past_key_values`, and EVERY read/write is
     guarded by `if cache_params is not None`. That is exactly the property the
     extra RVD passes rely on: setting `past_key_values=None, use_cache=False`
     (see `_kwargs_without_cache`) makes each extra pass a self-contained fresh
     scan over the current sequence — no KV appended, no conv/recurrent state
     read or mutated, for either mixer kind. Combined with the existing rule
     that recurrence only fires during multi-token prefill (T > 1) and never on
     the single-token decode step (T == 1, the DeltaNet per-step recurrence),
     the real cache the parent forward produced from the canonical pass is left
     byte-identical to a vanilla forward.

Strategy (unchanged): replace text_model.layers[block_start..block_end] with
shims that *subclass* the original Qwen3_5DecoderLayer so that isinstance
checks, GradientCheckpointingLayer inheritance, accelerate device hooks, and the
plain-Tensor return contract (Qwen3_5DecoderLayer.forward returns a bare
torch.Tensor) all Just Work.

The shims:
  - `RecurrentBlockHead` runs the full block [block_start..block_end] once
    canonically, then (depending on EVOLUTION_MODE) K-1 additional times on a
    subset of positions, then returns a single Tensor.
  - `IdentityShim` is a no-op forward (just returns hidden_states).

The originals live in `text_model._rvd_orig_layers` (plain Python list) and in a
hidden `nn.ModuleList` registered as `text_model._rvd_held_layers` so their
parameters stay attached to the model (needed for state_dict / device-placement
/ parameter ownership). The parent's layer loop iterates `text_model.layers`,
which no longer references the originals at the block positions — only via
`_rvd_held_layers` (which the parent never iterates).

K=1 (or mode="none") reduces to a single block pass through the original layers
and must byte-match the vanilla forward.

Public API (identical to the Qwen3-VL version):
    patch_model(model, block_start, block_end, tokenizer=None, mode=None) -> model
    set_K(model, K)
    get_K(model) -> int
    set_mode(model, mode)          # runtime override of EVOLUTION_MODE
    get_mode(model) -> str
    unpatch_model(model) -> model
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

# Qwen3.5 keeps the same visual-placeholder token strings as the Qwen3-VL family
# (the Qwen3.5 processor reuses the Qwen3-VL image/video processors). These are
# only used as an optional cross-check; the authoritative ids come from the
# model config (`image_token_id` / `video_token_id`), which always exist.
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
# =============================================================================

EVOLUTION_MODE = "all"

_VALID_MODES = ("vision", "all", "language", "vision_average", "none")

assert EVOLUTION_MODE in _VALID_MODES, (
    f"EVOLUTION_MODE={EVOLUTION_MODE!r} is not one of {_VALID_MODES}"
)


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


def _call_layer_hs(layer: nn.Module, hs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Call a decoder layer and return only the hidden_states Tensor.

    Qwen3_5DecoderLayer.forward returns a plain torch.Tensor, but we defensively
    unpack tuples in case a framework wrapper (e.g. a forward hook returning a
    modified output) wraps it.
    """
    out = layer(hs, *args, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


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
# Dynamic shim classes (built at patch time)
# -----------------------------------------------------------------------------

def _build_shim_classes(orig_layer_cls: type):
    """
    Build two subclasses of orig_layer_cls (== Qwen3_5DecoderLayer):
      - RecurrentBlockHead: replaces the layer at block_start; runs the full
        block K times internally.
      - IdentityShim: replaces layers (block_start, block_end]; no-op.

    Both subclass orig_layer_cls so that:
      * isinstance(shim, orig_layer_cls) is True. This is essential because
        transformers' _can_record_outputs target for hidden_states is exactly
        Qwen3_5DecoderLayer.
      * Any class-based wrapping (GradientCheckpointingLayer base, deprecation
        decorators, accelerate hooks attached by class, and the
        _no_split_modules=["Qwen3_5DecoderLayer", ...] treatment) is inherited.

    Neither shim calls super().__init__() with the original signature because we
    don't have the config easily; we initialize as a bare nn.Module and set just
    the attributes we need. The framework only ever looks at our `forward` and at
    isinstance() checks, both of which we satisfy.
    """

    class RecurrentBlockHead(orig_layer_cls):
        # IMPORTANT: we deliberately do NOT call super().__init__(config, layer_idx)
        # because (a) we don't have config conveniently and (b) we don't need any
        # of the parent's submodules — the *original* layers (held elsewhere) are
        # what actually runs. nn.Module.__init__ is enough to make this a valid
        # Module instance.
        def __init__(self, inner_layers_list, model_ref_holder):
            nn.Module.__init__(self)
            # Plain Python attributes (not submodules). The original layers live
            # in text_model._rvd_held_layers so PyTorch still owns them; here we
            # just keep references for the forward loop.
            object.__setattr__(self, "_inner_layers", inner_layers_list)
            object.__setattr__(self, "_model_ref_holder", model_ref_holder)
            # Marker for unpatch / introspection.
            self._rvd_is_head = True

        def _read_state(self):
            ref = self._model_ref_holder.get("model", None)
            if ref is None:
                return 1, None, "none"
            K = getattr(ref, "_rvd_K", 1)
            vmask = getattr(ref, "_rvd_current_vmask", None)
            mode = getattr(ref, "_rvd_mode", EVOLUTION_MODE)
            return K, vmask, mode

        def _one_block_pass(self, hs, args, kwargs):
            for layer in self._inner_layers:
                hs = _call_layer_hs(layer, hs, *args, **kwargs)
            return hs

        @staticmethod
        def _kwargs_without_cache(kwargs):
            """Strip cache-related kwargs for the extra RVD passes.

            For the extra passes we do NOT want to:
              * append new K,V (full-attention layers) or mutate the conv /
                recurrent delta-rule state (Gated DeltaNet linear-attention
                layers) — either would corrupt the cache the canonical pass
                built,
              * read previously-cached state as context (it was produced by the
                canonical pass on the same tokens; reading it would mean
                attending to / accumulating a "previous version of myself").

            In Qwen3_5DecoderLayer the cache is threaded in as
            `cache_params=past_key_values`, and both the full-attention path
            (`past_key_values.update(...)`) and the DeltaNet path
            (`cache_params.update_conv_state` / `.update_recurrent_state`, and
            the `has_previous_state` read) are all guarded by
            `if cache_params is not None`. So setting past_key_values=None and
            use_cache=False makes each extra pass a fresh, self-contained scan of
            just the current sequence for BOTH mixer kinds.
            """
            kw = dict(kwargs)
            kw["past_key_values"] = None
            kw["use_cache"] = False
            # Some attention implementations look at cache_position; drop it so
            # attention falls back to "compute from inputs_embeds shape".
            kw.pop("cache_position", None)
            return kw

        def _active_mask(self, mode, vmask, canonical):
            """Return the [B, T] bool mask of positions that EVOLVE, or None to
            signal "skip recurrence, return canonical".

            All tensors are built/moved onto canonical.device so the subsequent
            torch.where / block passes stay on one shard.
            """
            B, T = canonical.shape[0], canonical.shape[1]
            dev = canonical.device

            if mode == "all":
                # Every position evolves. No vision mask required (works even
                # when only inputs_embeds were provided).
                return torch.ones((B, T), dtype=torch.bool, device=dev)

            # vision / language / vision_average all need the vision mask.
            if vmask is None:
                return None
            vloc = vmask.to(dev)

            if mode in ("vision", "vision_average"):
                # Evolve vision positions; require at least one to exist.
                return vloc if bool(vloc.any().item()) else None

            if mode == "language":
                # Evolve everything that is NOT a vision pad token.
                lang = ~vloc
                return lang if bool(lang.any().item()) else None

            return None

        def forward(self, hidden_states, *args, **kwargs):
            K, vmask, mode = self._read_state()

            # Canonical pass (always). Uses the real (hybrid KV + recurrent)
            # cache, so downstream generation sees the same cache state it would
            # after a vanilla forward through this block.
            canonical = self._one_block_pass(hidden_states, args, kwargs)

            # Pure baseline: no recurrence at all.
            if mode == "none" or K == 1:
                return canonical

            T = canonical.shape[1]

            # Incremental (cached) decoding => hidden_states.shape[1] == 1.
            # Recurrence only fires during multi-token prefill. This also keeps
            # us off the DeltaNet single-token recurrence path entirely.
            if T == 1:
                return canonical

            # If a vision mask exists it must align with this pass (i.e. this is
            # the full-prompt prefill, not a chunk / cached step). vmask is built
            # from the *full* input_ids in the pre-forward hook, so a mismatch
            # means we can't line vision positions up with `canonical`.
            if vmask is not None and vmask.shape[1] != T:
                return canonical

            # Which positions evolve (mode-dependent).
            active = self._active_mask(mode, vmask, canonical)
            if active is None:
                return canonical
            active_b = active.unsqueeze(-1)  # [B, T, 1]

            # Extra passes: no cache reads/writes (see _kwargs_without_cache).
            iter_kwargs = self._kwargs_without_cache(kwargs)

            if mode == "vision_average":
                # Read back the MEAN over all K passes (canonical is pass #1),
                # not just the final iterate. Feedback loop is identical to
                # "vision" — only the read-out differs.
                acc = canonical
                x_iter = canonical
                for _ in range(1, K):
                    inp = torch.where(active_b, x_iter, canonical)
                    x_iter = self._one_block_pass(inp, args, iter_kwargs)
                    acc = acc + x_iter
                mean = acc / float(K)
                return torch.where(active_b, mean, canonical)

            # Default read-back (vision / language / all): the final iterate at
            # the active positions, canonical everywhere else.
            x_iter = canonical
            for _ in range(1, K):
                inp = torch.where(active_b, x_iter, canonical)
                x_iter = self._one_block_pass(inp, args, iter_kwargs)
            return torch.where(active_b, x_iter, canonical)

    class IdentityShim(orig_layer_cls):
        def __init__(self):
            nn.Module.__init__(self)
            self._rvd_is_identity = True

        def forward(self, hidden_states, *args, **kwargs):
            # Qwen3_5DecoderLayer.forward returns a plain Tensor. We mirror that.
            return hidden_states

    return RecurrentBlockHead, IdentityShim


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def patch_model(model, block_start: int, block_end: int, tokenizer=None,
                mode: Optional[str] = None):
    text_model = _get_text_model(model)
    L = len(text_model.layers)
    assert 0 <= block_start <= block_end < L, (
        f"block range [{block_start}, {block_end}] out of bounds for {L} layers"
    )

    # Resolve the image/video placeholder token ids.
    image_pad_id, video_pad_id = _resolve_pad_ids(model, tokenizer)
    model._rvd_image_pad_id = image_pad_id
    model._rvd_video_pad_id = video_pad_id if video_pad_id is not None else -1
    model._rvd_K = 1
    model._rvd_current_vmask = None

    # Evolution mode: explicit arg wins, else the module-level default.
    chosen_mode = EVOLUTION_MODE if mode is None else mode
    if chosen_mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown evolution mode {chosen_mode!r}; valid: {_VALID_MODES}"
        )
    model._rvd_mode = chosen_mode

    ref_holder = {"model": model}

    # Snapshot original layers (idempotent across repeated patch_model calls).
    if not hasattr(text_model, "_rvd_orig_layers"):
        text_model._rvd_orig_layers = list(text_model.layers)
        # Hidden ModuleList so PyTorch still owns the parameters of the originals
        # after we swap them out of text_model.layers. The parent's forward never
        # iterates this name, so no double-execution.
        text_model._rvd_held_layers = nn.ModuleList(text_model._rvd_orig_layers)

    text_model._rvd_block_start = block_start
    text_model._rvd_block_end = block_end

    # Build shim classes from the actual decoder-layer class in use
    # (Qwen3_5DecoderLayer — the same class for both linear and full layers).
    orig_layer_cls = type(text_model._rvd_orig_layers[0])
    RecurrentBlockHeadCls, IdentityShimCls = _build_shim_classes(orig_layer_cls)

    inner_layers_list: List[nn.Module] = [
        text_model._rvd_orig_layers[i] for i in range(block_start, block_end + 1)
    ]

    head = RecurrentBlockHeadCls(inner_layers_list, ref_holder)

    new_layers = nn.ModuleList()
    for i in range(L):
        if i == block_start:
            new_layers.append(head)
        elif block_start < i <= block_end:
            new_layers.append(IdentityShimCls())
        else:
            new_layers.append(text_model._rvd_orig_layers[i])
    text_model.layers = new_layers

    # Move shims to the right device/dtype so device_map='auto' setups still
    # work. We piggy-back on whatever device the surrounding original layers
    # are on.
    _align_shims_to_device(text_model, block_start, block_end, L)

    if not hasattr(model, "_rvd_hook_handle"):
        h = model.register_forward_pre_hook(
            _make_pre_forward_hook(model), with_kwargs=True
        )
        model._rvd_hook_handle = h

    return model


def _align_shims_to_device(text_model, block_start, block_end, L):
    """Place each shim on the device of its corresponding ORIGINAL layer.

    Under device_map='auto' the layers in [block_start..block_end] may span
    multiple GPUs. We want each shim to live on the same device as the original
    layer it replaces, so:
      - RecurrentBlockHead at block_start sits on the device of orig[block_start]
        (its forward will be entered with hidden_states already on that device,
        since the parent's previous layer ran on the same shard or accelerate
        inserted a transfer).
      - Each IdentityShim mirrors its corresponding original layer's device.
    The original layers themselves still live on their assigned devices (held
    under text_model._rvd_held_layers), so when RecurrentBlockHead calls them
    sequentially, accelerate's per-module device hooks will transfer
    hidden_states between GPUs as needed — exactly as in the vanilla forward.
    """
    def _device_of(layer):
        try:
            return next(layer.parameters()).device
        except StopIteration:
            return None

    for i in range(block_start, block_end + 1):
        dev = _device_of(text_model._rvd_orig_layers[i])
        if dev is not None:
            text_model.layers[i].to(device=dev)


def unpatch_model(model):
    text_model = _get_text_model(model)
    if hasattr(text_model, "_rvd_orig_layers"):
        text_model.layers = nn.ModuleList(text_model._rvd_orig_layers)
        del text_model._rvd_orig_layers
    if hasattr(text_model, "_rvd_held_layers"):
        del text_model._rvd_held_layers
    for attr in ("_rvd_block_start", "_rvd_block_end"):
        if hasattr(text_model, attr):
            delattr(text_model, attr)
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
    """Runtime override of EVOLUTION_MODE for a single patched model."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown evolution mode {mode!r}; valid: {_VALID_MODES}")
    model._rvd_mode = mode


def get_mode(model) -> str:
    return getattr(model, "_rvd_mode", EVOLUTION_MODE)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_text_model(model: nn.Module) -> nn.Module:
    # Qwen3.5 (Qwen3_5ForConditionalGeneration) exposes the text backbone at
    # model.model.language_model (a Qwen3_5TextModel), alongside the vision tower
    # model.model.visual. We keep robust fallbacks in case a checkpoint/wrapper
    # nests things differently.
    if hasattr(model, "model") and hasattr(model.model, "language_model") \
            and hasattr(model.model.language_model, "layers"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    # Some wrappers name the text submodule `text_model`.
    if hasattr(model, "model") and hasattr(model.model, "text_model") \
            and hasattr(model.model.text_model, "layers"):
        return model.model.text_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError(
        "Could not locate text model with .layers; this file targets Qwen3.5 "
        "(expected model.model.language_model.layers)."
    )


def _resolve_pad_ids(model, tokenizer) -> Tuple[int, Optional[int]]:
    # The model config is authoritative for Qwen3.5: `image_token_id` /
    # `video_token_id` are the exact placeholder ids the model uses to scatter
    # vision features into the sequence, so masking on them lines up with the
    # hidden states 1:1. We use the config first, and only fall back to a
    # tokenizer string lookup if the config doesn't carry them.
    cfg = getattr(model, "config", None)
    image_pad_id = getattr(cfg, "image_token_id", None) if cfg is not None else None
    video_pad_id = getattr(cfg, "video_token_id", None) if cfg is not None else None

    if image_pad_id is not None:
        return image_pad_id, video_pad_id

    if tokenizer is not None:
        tok_img = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        tok_vid = tokenizer.convert_tokens_to_ids(VIDEO_PAD_TOKEN)
        if tok_img is not None and tok_img != tokenizer.unk_token_id:
            return tok_img, (None if tok_vid == tokenizer.unk_token_id else tok_vid)

    raise RuntimeError(
        "Could not resolve the <|image_pad|> token id from config.image_token_id "
        "or the tokenizer."
    )