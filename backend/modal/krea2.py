"""
krea2-identity-edit.py — Krea 2 image generation + reference-image identity editing for
Aiko, on Modal. App: oppa-ai-org--aiko-imagegen-krea2

============================================================================================
A10G ADAPTATION — READ THIS FIRST
============================================================================================
The original version of this script assumed A100/H100 and kept pipe_raw AND pipe_turbo
fully resident on GPU in bf16, plus a separate Qwen3-VL-4B for grounding. That's 50-60GB+
of weights alone — nowhere close to fitting A10G's 24GB.

This version fits A10G by doing three things differently:

  1. NF4 QUANTIZATION on every heavy weight (both transformers + the VLM), via
     bitsandbytes' BitsAndBytesConfig. Rough sizes: a ~12B-class transformer in bf16 is
     ~24GB; in NF4 it's ~6-7GB. The 4B VLM goes from ~8GB (bf16) to ~2.5GB (NF4).

  2. TIME-SHARING THE GPU instead of keeping everything resident. At rest, every model
     lives on CPU (quantized weights still take real CPU RAM — see the memory bump on
     the @app.cls below). Each call moves only the piece it currently needs onto "cuda",
     runs, then evicts it and calls torch.cuda.empty_cache() before the next piece comes
     on. Concretely, identity_edit() runs in two GPU phases:
       Phase A: VLM on GPU  -> compute pos_embeds (+ neg_embeds if CFG>1) -> VLM off GPU
       Phase B: active diffusion pipe on GPU -> denoise loop + VAE decode -> pipe off GPU
     These phases never overlap on GPU, so peak VRAM is roughly max(VLM, one pipe) instead
     of VLM + both pipes.

  3. SDPA ATTENTION + TIGHTER DEFAULT RESOLUTION for the edit path specifically.
     generate() (plain txt2img) was fine on A10G, but identity_edit()'s custom
     krea2_edit_forward() concatenates [text, source latent(s), target latent] into one
     long sequence before running the transformer blocks. With flash-attn disabled
     (DIFFUSERS_NO_FLASH_ATTN=1) this fell back to eager/naive attention, whose memory
     scales quadratically with sequence length — one or two extra source-image token
     blocks on top of the target was enough to blow the 24GB budget mid-denoise, even
     though the exact same GPU handled generate() fine. Fix applied below:
       - attn_implementation="sdpa" on both the transformer and text_encoder loads,
         which routes attention through PyTorch's memory-efficient SDPA kernel instead
         of eager bmm attention, without needing the separate flash-attn package.
       - max_pixels default lowered from 2,000,000 -> 1,000,000 and exposed as a
         request parameter, so sequence length (and therefore attention memory) is
         smaller by default and tunable per-call.
       - grounding_px default lowered from 768 -> 512 for the same reason on the VLM
         (Phase A) side.
     If you still OOM after this, drop max_pixels further (e.g. 600_000) before
     touching anything else — it's the cheapest remaining lever.

  4. OOM-SAFE ERROR HANDLING. Previously an OOM inside identity_edit() (or during
     load()) would SIGKILL the whole container mid-request, so the caller (curl) got
     a closed connection with an EMPTY body — which is why `json.load` on the client
     side failed with "Expecting value: line 1 column 1 (char 0)" instead of a useful
     error. identity_edit() and generate() now catch torch.cuda.OutOfMemoryError,
     evict whatever's on GPU, clear the cache, and raise an HTTPException with a
     real message and a 507 status instead of dying silently. This won't save you
     from an OOM (you still need the fixes above/below for that), but it turns a
     confusing empty-body crash into an actual diagnosable error message.

  I have NOT been able to test this against real weights/a real GPU (no GPU access in
  the sandbox that wrote this), so treat the NF4 memory estimates above as engineering
  estimates, not measurements. If you still OOM on first run, the first things to try
  are: lowering `grounding_px`, capping `max_pixels` further, and confirming
  `attn_implementation="sdpa"` is actually being honored by your installed
  diffusers/transformers versions (print `pipe.transformer.config._attn_implementation`
  after load to check).

Everything below the quantization/offload/attention changes (RoPE-frame source
injection, LoRA caveats, image-grounded text conditioning template) is unchanged from
the original — see inline comments there for the same caveats about unverified LoRA
key names and tokenization offsets.

Requires Modal secrets:
  - huggingface-secret (HF_TOKEN) — the LoRA repo is gated behind the Krea 2 Community
    License click-through same as the base models, so you need a token with access.
"""

import io
import base64
import modal

# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "git+https://github.com/huggingface/transformers.git",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "Pillow",
        "numpy",
        "einops",
        "peft",
        "bitsandbytes",       # needed for NF4 quantization
        "fastapi[standard]",
    )
    .env({
        "DIFFUSERS_NO_FLASH_ATTN": "1",
        # Reduces allocator fragmentation — cheap extra headroom on a GPU this
        # tight on memory. Suggested directly by the OOM error text itself.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

weights_volume = modal.Volume.from_name("aiko-imagegen-krea2-weights", create_if_missing=True)
lora_volume = modal.Volume.from_name("aiko-imagegen-krea2-identity-lora", create_if_missing=True)
WEIGHTS_DIR = "/weights"
LORA_DIR = "/lora"

RAW_MODEL_ID = "krea/Krea-2-Raw"      # undistilled — use for removals (CFG 3, ~20 steps)
TURBO_MODEL_ID = "krea/Krea-2-Turbo"  # distilled — use for most edits (CFG 1, 8-12 steps)
LORA_REPO_ID = "conradlocke/krea2-identity-edit"
LORA_FILENAME = "krea2_identity_edit_v1_1.safetensors"  # full-rank; see repo for r128/r64 variants
VLM_PROCESSOR_ID = "Qwen/Qwen3-VL-4B-Instruct"  # for image-grounded text conditioning only

# NEW: default caps for the edit path, tuned down from the original A100-era defaults
# so identity_edit()'s longer [text + source(s) + target] attention sequence fits in
# A10G's 24GB when combined with attn_implementation="sdpa". Both are still overridable
# per-request via EditRequest.
DEFAULT_MAX_PIXELS = 1_000_000   # was 2,000,000
DEFAULT_GROUNDING_PX = 512       # was 768

app = modal.App("aiko-imagegen-krea2", image=image)


# ---------------------------------------------------------------------------
# one-time weight downloader — run with:
#   modal run krea2-identity-edit.py::download_weights
# (unchanged from original — this step doesn't touch GPU memory)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: weights_volume, LORA_DIR: lora_volume},
    timeout=3600,
)
def download_weights(model: str = "both"):
    import os
    from huggingface_hub import snapshot_download, hf_hub_download

    hf_token = os.environ["HF_TOKEN"]

    if model in ("raw", "both"):
        print("Downloading Krea 2 Raw weights...")
        snapshot_download(
            RAW_MODEL_ID,
            local_dir=f"{WEIGHTS_DIR}/krea2-raw",
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5"],
        )
        weights_volume.commit()

    if model in ("turbo", "both"):
        print("Downloading Krea 2 Turbo weights...")
        snapshot_download(
            TURBO_MODEL_ID,
            local_dir=f"{WEIGHTS_DIR}/krea2-turbo",
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5"],
        )
        weights_volume.commit()

    print("Downloading Krea 2 Identity Edit LoRA (v1.1)...")
    hf_hub_download(
        LORA_REPO_ID,
        filename=LORA_FILENAME,
        local_dir=LORA_DIR,
        token=hf_token,
    )
    lora_volume.commit()
    print("Done. Volumes committed.")


# ---------------------------------------------------------------------------
# RoPE-frame source injection — ported from lbouaraba/comfyui-krea2edit's
# krea2_edit_forward(), rewritten against diffusers' Krea2Transformer2DModel
# submodules (img_in / time_embed / time_mod_proj / text_fusion / txt_in /
# rotary_emb / transformer_blocks / final_layer). UNCHANGED from the original —
# this is pure math and doesn't care which device the tensors live on, and it
# doesn't care whether the blocks underneath are running eager or SDPA attention
# either — that's decided by each block's own attn_implementation, set at
# from_pretrained() time in load() below.
# ---------------------------------------------------------------------------
def _build_position_ids(text_len, grid_h, grid_w, num_sources, device):
    """(text | source_1..N | target) position ids, shape (seq, 3).
    Text sits at the all-zero origin (matches Krea2Pipeline.prepare_position_ids).
    Sources get frame index 1..N (RoPE axis 0); the noisy target keeps frame 0 — the
    same convention Krea2Pipeline already uses for plain text-to-image, so this is a
    strict superset, not a different scheme."""
    import torch

    text_ids = torch.zeros(text_len, 3, device=device)

    def grid_ids(frame):
        ids = torch.zeros(grid_h, grid_w, 3, device=device)
        ids[..., 0] = frame
        ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        return ids.reshape(grid_h * grid_w, 3)

    source_ids = [grid_ids(i + 1) for i in range(num_sources)]
    target_ids = grid_ids(0)
    return torch.cat([text_ids] + source_ids + [target_ids], dim=0)


def krea2_edit_forward(transformer, target_hidden_states, source_latents, text_embeds,
                        timestep, grid_h, grid_w, text_attention_mask=None):
    """Krea2Transformer2DModel.forward, but with clean source-latent block(s) prepended
    before the noisy target. Mirrors diffusers' own forward exactly except for the extra
    concatenation/slice around the source tokens.

    target_hidden_states : (B, tgt_len, in_channels) packed noisy target latent
    source_latents        : list of (B, src_len, in_channels) packed CLEAN source latents
                             (already patch-packed, same grid_h/grid_w as the target)
    text_embeds            : (B, text_len, num_text_layers, text_hidden_dim) tapped text states
    timestep               : (B,) flow-matching time in [0,1]
    """
    import torch
    import torch.nn.functional as F

    m = transformer
    bs = target_hidden_states.shape[0]
    text_len = text_embeds.shape[1]
    tgt_len = target_hidden_states.shape[1]
    src_lens = [s.shape[1] for s in source_latents]

    temb = m.time_embed(timestep, dtype=target_hidden_states.dtype)
    temb_mod = m.time_mod_proj(F.gelu(temb, approximate="tanh"))

    text_mask_4d = None
    attn_mask = None
    if text_attention_mask is not None:
        text_mask_4d = text_attention_mask[:, None, None, :]
        img_len = sum(src_lens) + tgt_len
        img_mask = text_attention_mask.new_ones((bs, img_len))
        attn_mask = torch.cat([text_attention_mask, img_mask], dim=1)[:, None, None, :]

    encoder_hidden_states = m.text_fusion(text_embeds, attention_mask=text_mask_4d)
    encoder_hidden_states = m.txt_in(encoder_hidden_states)

    tgt_embedded = m.img_in(target_hidden_states)
    src_embedded = [m.img_in(s) for s in source_latents]

    combined = torch.cat([encoder_hidden_states] + src_embedded + [tgt_embedded], dim=1)

    position_ids = _build_position_ids(text_len, grid_h, grid_w, len(source_latents), combined.device)
    image_rotary_emb = m.rotary_emb(position_ids)

    for block in m.transformer_blocks:
        combined = block(combined, temb_mod, image_rotary_emb, attn_mask)

    # Drop text (matches diffusers' own forward), run the final layer over the
    # remaining (sources + target) tokens, then keep only the target segment —
    # mirrors ai-toolkit's predict_velocity_edit / the comfy node's final slice.
    combined = combined[:, text_len:]
    out = m.final_layer(combined, temb)
    out = out[:, -tgt_len:, :]
    return out


# ---------------------------------------------------------------------------
# GPU time-sharing helpers — move a nn.Module (or a whole pipeline's heavy
# submodules) between "cuda" and "cpu" on demand, so only one heavy model
# occupies GPU memory at any given instant.
# ---------------------------------------------------------------------------
def _pipe_to(pipe, device):
    """Move a diffusion pipeline's heavy submodules to `device`. Light stuff
    (schedulers, processors, config objects) has no meaningful memory footprint
    and is left alone."""
    for attr in ("transformer", "vae", "text_encoder"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            mod.to(device)


def _sync_and_clear():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _evict_all(pipe):
    """Best-effort emergency eviction used in OOM except-blocks below — moves
    everything for the given pipe back to CPU and clears the cache. Wrapped in
    its own try/except since we may be calling this while already handling an
    OOM, and a second CUDA call can itself raise."""
    try:
        _pipe_to(pipe, "cpu")
    except Exception:
        pass
    try:
        _sync_and_clear()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# model class
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    # Both pipelines + the VLM now live quantized on CPU/disk between GPU
    # phases instead of being reloaded per call, so we ask for more host RAM
    # than the default. Adjust down if your Modal plan caps this.
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: weights_volume, LORA_DIR: lora_volume},
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class AikoKrea2ImageGen:

    @modal.enter()
    def load(self):
        import os
        import torch
        from diffusers import Krea2Pipeline
        from transformers import AutoProcessor, BitsAndBytesConfig

        # NF4 config shared by both transformers and the VLM. double_quant
        # squeezes a bit more out of an already-quantized tensor at negligible
        # quality cost — worth it here since we're memory-constrained, not
        # compute-constrained.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        def _load(local_path, model_id):
            if not os.path.exists(f"{local_path}/transformer"):
                from huggingface_hub import snapshot_download
                print(f"{model_id} weights missing — downloading...")
                snapshot_download(
                    model_id, local_dir=local_path, token=os.environ["HF_TOKEN"],
                    ignore_patterns=["*.msgpack", "*.h5"],
                )
                weights_volume.commit()
            # Load the transformer NF4-quantized explicitly, then hand it to the
            # pipeline — quantization_config on from_pretrained's top level doesn't
            # reliably propagate to every submodule for custom pipeline classes.
            #
            # NEW: attn_implementation="sdpa" routes attention through PyTorch's
            # memory-efficient scaled-dot-product-attention kernel instead of eager
            # bmm attention. This matters a lot more for identity_edit()'s custom
            # krea2_edit_forward() than for plain generate() — the edit path's
            # sequence is [text, source latent(s), target latent] concatenated
            # together, which is considerably longer than text2img's, and eager
            # attention's memory scales quadratically with that length. This was
            # the likely cause of edit-only OOMs on A10G even though generate()
            # was fine.
            from diffusers import Krea2Transformer2DModel
            transformer = Krea2Transformer2DModel.from_pretrained(
                local_path, subfolder="transformer",
                quantization_config=bnb_config, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )

            # NF4-quantize the text_encoder (Qwen3-VL, ~8.3GB in bf16) too, and
            # also give it SDPA for the same reason as above — grounding runs the
            # VLM over the instruction text plus one or two resized reference
            # images, which is a much longer sequence than a plain caption.
            # AutoModel is used here (rather than a specific class) so this
            # doesn't depend on guessing Qwen3-VL's exact registered class name;
            # trust_remote_code is on since Qwen-VL family models commonly need it.
            from transformers import AutoModel
            text_encoder = AutoModel.from_pretrained(
                local_path, subfolder="text_encoder",
                quantization_config=bnb_config, torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )

            pipe = Krea2Pipeline.from_pretrained(
                local_path, transformer=transformer, text_encoder=text_encoder,
                torch_dtype=torch.bfloat16,
            )
            # Everything parked on CPU at rest — moved to cuda only during its
            # active phase inside generate()/identity_edit().
            _pipe_to(pipe, "cpu")
            return pipe

        self.pipe_raw = _load(f"{WEIGHTS_DIR}/krea2-raw", RAW_MODEL_ID)
        self.pipe_turbo = _load(f"{WEIGHTS_DIR}/krea2-turbo", TURBO_MODEL_ID)

        lora_path = f"{LORA_DIR}/{LORA_FILENAME}"
        if not os.path.exists(lora_path):
            from huggingface_hub import hf_hub_download
            print("Identity-edit LoRA missing — downloading...")
            hf_hub_download(LORA_REPO_ID, filename=LORA_FILENAME, local_dir=LORA_DIR,
                             token=os.environ["HF_TOKEN"])
            lora_volume.commit()

        # NOTE: loading LoRA onto an already-NF4-quantized transformer works with
        # peft/diffusers' merge-free adapter path (weights stay in the base
        # dtype, LoRA deltas ride alongside), but has NOT been verified against
        # this specific model. If load_lora_weights errors on a quantized base,
        # the fallback path below still gives you the key-overlap diagnostic.
        self.load_identity_lora(self.pipe_raw, lora_path)
        self.load_identity_lora(self.pipe_turbo, lora_path)

        # VLM: also NF4-quantized, also parked on CPU at rest.
        self.vlm_processor = AutoProcessor.from_pretrained(VLM_PROCESSOR_ID)
        from transformers import AutoModel
        self.vlm_text_encoder = None  # lazily bound per-pipe text_encoder below

        print("Krea 2 Raw + Turbo (NF4, SDPA) + Identity Edit LoRA ready. Everything parked on CPU.")

    def load_identity_lora(self, pipe, lora_path, adapter_name="identity_edit", scale=1.0):
        """Load the community identity-edit LoRA onto a pipeline's transformer.
        Tries the standard diffusers path first; on key mismatch, prints an overlap
        report instead of failing silently, since the LoRA's state-dict naming vs.
        diffusers' PEFT naming hasn't been verified (see module docstring)."""
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipe.set_adapters([adapter_name], adapter_weights=[scale])
            print(f"LoRA loaded cleanly onto {pipe.__class__.__name__}.")
        except Exception as e:
            import safetensors.torch as st
            lora_keys = set(st.load_file(lora_path).keys())
            model_keys = set(pipe.transformer.state_dict().keys())
            sample_lora = sorted(lora_keys)[:8]
            print(
                f"[LoRA load failed on {pipe.__class__.__name__}] {e}\n"
                f"Sample LoRA keys: {sample_lora}\n"
                f"This almost always means the .safetensors uses ComfyUI/BFL-style key "
                f"names (e.g. 'diffusion_model.blocks.N...') rather than diffusers' "
                f"'transformer_blocks.N...' naming, OR that the quantized base transformer "
                f"needs the adapter attached before quantization rather than after. You'll "
                f"likely need a small key-remap dict before calling "
                f"pipe.transformer.load_lora_adapter() directly — compare `sample_lora` "
                f"above against `sorted(model_keys)[:8]` ({sorted(model_keys)[:8]})."
            )
            raise

    # -----------------------------------------------------------------
    def _encode_source_latent(self, pipe, pil_image, height, width):
        """VAE-encode a reference image with NO added noise — this is a clean in-context
        token block, not an SDEdit starting point. Assumes pipe.vae is currently on cuda
        (caller's responsibility — see generate()/identity_edit() phase management)."""
        import torch

        image_tensor = pipe.image_processor.preprocess(pil_image, height=height, width=width)
        image_tensor = image_tensor.to(device="cuda", dtype=pipe.vae.dtype).unsqueeze(2)

        with torch.no_grad():
            raw_latents = pipe.vae.encode(image_tensor).latent_dist.mode()

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1).to(raw_latents.device, raw_latents.dtype)
        latents_std = (1.0 / torch.tensor(pipe.vae.config.latents_std)).view(
            1, pipe.vae.config.z_dim, 1, 1, 1).to(raw_latents.device, raw_latents.dtype)
        norm_latents = ((raw_latents - latents_mean) * latents_std).squeeze(2)

        b, c, h, w = norm_latents.shape
        return pipe._pack_latents(norm_latents, b, c, h, w)

    KREA2_EDIT_PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    KREA2_EDIT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

    def _grounded_text_embeds(self, pipe, prompt, pil_images, grounding_px=DEFAULT_GROUNDING_PX):
        """Image-grounded instruction encoding: Qwen3-VL reads the source image(s) WHILE
        reading the instruction, matching how the LoRA was trained. Caller is responsible
        for pipe.text_encoder currently being on cuda — see phase management below."""
        import torch

        imgs = []
        for im in pil_images:
            w, h = im.size
            if grounding_px and max(w, h) > grounding_px:
                s = grounding_px / max(w, h)
                im = im.resize((round(w * s), round(h * s)))
            imgs.append(im.convert("RGB"))

        text = self.KREA2_EDIT_PREFIX + (self.VISION_BLOCK * len(imgs)) + prompt + self.KREA2_EDIT_SUFFIX

        proc = self.vlm_processor(
            text=[text], images=imgs, return_tensors="pt", return_mm_token_type_ids=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = pipe.text_encoder(
                input_ids=proc["input_ids"],
                attention_mask=proc.get("attention_mask"),
                pixel_values=proc.get("pixel_values"),
                image_grid_thw=proc.get("image_grid_thw"),
                mm_token_type_ids=proc.get("mm_token_type_ids"),
                output_hidden_states=True,
            )

        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in pipe.text_encoder_select_layers], dim=2
        )
        prefix_idx = pipe.prompt_template_encode_start_idx  # 34 — fixed prefix length
        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = proc["attention_mask"][:, prefix_idx:].bool()
        # Move embeds off GPU with the rest of text_encoder's phase — cheap tensors,
        # kept resident is fine, but we still detach so they don't pin a graph.
        return hidden_states.detach(), attention_mask.detach()

    # -----------------------------------------------------------------
    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 52,
        guidance_scale: float = 3.5,
        seed: int = -1,
        negative_prompt: str = "",
        max_sequence_length: int = 512,
    ) -> str:
        """Plain text-to-image, no reference. Always uses Raw (undistilled) for full CFG."""
        import torch
        from fastapi import HTTPException

        pipe = self.pipe_raw
        _pipe_to(pipe, "cuda")
        try:
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
            image = pipe(
                prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length, generator=generator,
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:
            _evict_all(pipe)
            raise HTTPException(
                status_code=507,
                detail=f"GPU out of memory during generate() at {width}x{height}, "
                       f"{steps} steps. Try a smaller width/height. ({e})",
            )
        finally:
            _pipe_to(pipe, "cpu")
            _sync_and_clear()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @modal.method()
    def identity_edit(
        self,
        instruction: str,
        reference_image: str,               # base64 PNG/JPG, the source to edit
        reference_image_b: str | None = None,  # optional 2nd ref (scene=a, subject=b)
        is_removal: bool = False,            # True -> Raw/CFG3/~20 steps per LoRA card
        steps: int | None = None,
        guidance_scale: float | None = None,
        grounding_px: int = DEFAULT_GROUNDING_PX,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        seed: int = -1,
    ) -> str:
        """Instruction-based, identity-preserving edit using the community LoRA's
        RoPE-frame source injection + image-grounded text conditioning.

        A10G phase management: Phase A brings up only the active pipe's text_encoder
        (the VLM) to compute pos/neg embeds, then evicts it. Phase B brings up the
        active pipe's transformer + vae to run the denoise loop + decode, then evicts
        those too. The two phases never overlap on GPU.

        max_pixels and grounding_px are now request-level knobs (defaulted lower than
        the original A100-era version) since this path's attention sequence length —
        and therefore memory — scales with both. If you hit an OOM here, lower
        max_pixels first (cheapest, biggest effect), then grounding_px, before anything
        else.
        """
        import torch
        import numpy as np
        from PIL import Image
        from fastapi import HTTPException

        pipe = self.pipe_raw if is_removal else self.pipe_turbo
        if steps is None:
            steps = 20 if is_removal else 10
        if guidance_scale is None:
            guidance_scale = 3.0 if is_removal else 1.0
        do_cfg = guidance_scale > 1.0

        ref_img = Image.open(io.BytesIO(base64.b64decode(reference_image))).convert("RGB")
        ref_img_b = None
        if reference_image_b is not None:
            ref_img_b = Image.open(io.BytesIO(base64.b64decode(reference_image_b))).convert("RGB")

        # Match output AR to the (first) source image, capped at `max_pixels`.
        # On A10G, if you still see OOMs on the denoise loop, lower max_pixels first —
        # this is the cheapest knob before touching grounding_px or steps.
        src_w, src_h = ref_img.size
        multiple = pipe.vae_scale_factor * pipe.patch_size
        scale = min(1.0, (max_pixels / (src_w * src_h)) ** 0.5)
        width = max(multiple, round(src_w * scale / multiple) * multiple)
        height = max(multiple, round(src_h * scale / multiple) * multiple)

        generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None

        # ---------------- Phase A: VLM grounding only ----------------
        pipe.text_encoder.to("cuda")
        try:
            ref_images_for_grounding = [ref_img] + ([ref_img_b] if ref_img_b is not None else [])
            pos_embeds, pos_mask = self._grounded_text_embeds(pipe, instruction, ref_images_for_grounding, grounding_px)
            neg_embeds = neg_mask = None
            if do_cfg:
                neg_embeds, neg_mask = self._grounded_text_embeds(pipe, "", ref_images_for_grounding, grounding_px)
        except torch.cuda.OutOfMemoryError as e:
            _evict_all(pipe)
            raise HTTPException(
                status_code=507,
                detail=f"GPU out of memory during VLM grounding (Phase A) at "
                       f"grounding_px={grounding_px}. Try a lower grounding_px. ({e})",
            )
        finally:
            pipe.text_encoder.to("cpu")
            _sync_and_clear()

        # ---------------- Phase B: transformer + vae only ----------------
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        try:
            pos_embeds = pos_embeds.to("cuda")
            pos_mask = pos_mask.to("cuda")
            if do_cfg:
                neg_embeds = neg_embeds.to("cuda")
                neg_mask = neg_mask.to("cuda")

            # --- source latent(s): clean, no noise, resized to the target grid ---
            source_latents = [self._encode_source_latent(pipe, ref_img, height, width)]
            if ref_img_b is not None:
                source_latents.append(self._encode_source_latent(pipe, ref_img_b, height, width))

            # --- target: pure noise, standard packed-latent shape ---
            num_channels_latents = pipe.transformer.config.in_channels // (pipe.patch_size ** 2)
            latents = pipe.prepare_latents(
                1, num_channels_latents, height, width, pos_embeds.dtype, "cuda", generator,
            )
            grid_h = height // (pipe.vae_scale_factor * pipe.patch_size)
            grid_w = width // (pipe.vae_scale_factor * pipe.patch_size)

            # --- timesteps: fixed mu=1.15 for Turbo (distilled), resolution-aware for Raw ---
            sigmas = np.linspace(1.0, 1 / steps, steps)
            if pipe.config.is_distilled:
                mu = 1.15
            else:
                from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
                mu = calculate_shift(
                    grid_h * grid_w,
                    pipe.scheduler.config.get("base_image_seq_len", 256),
                    pipe.scheduler.config.get("max_image_seq_len", 6400),
                    pipe.scheduler.config.get("base_shift", 0.5),
                    pipe.scheduler.config.get("max_shift", 1.15),
                )
            pipe.scheduler.set_timesteps(steps, device="cuda", sigmas=sigmas, mu=mu)
            pipe.scheduler.set_begin_index(0)
            timesteps = pipe.scheduler.timesteps

            with torch.no_grad():
                for t in timesteps:
                    timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = krea2_edit_forward(
                        pipe.transformer, latents, source_latents, pos_embeds, timestep,
                        grid_h, grid_w, text_attention_mask=pos_mask,
                    )

                    if do_cfg:
                        neg_noise_pred = krea2_edit_forward(
                            pipe.transformer, latents, source_latents, neg_embeds, timestep,
                            grid_h, grid_w, text_attention_mask=neg_mask,
                        )
                        noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # --- decode --- (wrapped in no_grad — this was previously outside the
            # loop's no_grad scope, which left the decoded tensor requiring grad
            # and broke postprocess()'s .numpy() call)
            with torch.no_grad():
                latents = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
                latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
                    1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
                latents_std = (1.0 / torch.tensor(pipe.vae.config.latents_std)).view(
                    1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
                pil_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        except torch.cuda.OutOfMemoryError as e:
            _evict_all(pipe)
            raise HTTPException(
                status_code=507,
                detail=f"GPU out of memory during denoise/decode (Phase B) at "
                       f"{width}x{height} ({grid_h}x{grid_w} grid, {steps} steps, "
                       f"{len(source_latents) if 'source_latents' in locals() else '?'} source "
                       f"image(s)). Try a lower max_pixels (currently {max_pixels}), fewer "
                       f"steps, or a single reference image. ({e})",
            )
        finally:
            pipe.transformer.to("cpu")
            pipe.vae.to("cpu")
            _sync_and_clear()

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# FastAPI wrapper
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

web_app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 52
    guidance_scale: float = 3.5
    seed: int = -1


class EditRequest(BaseModel):
    instruction: str
    reference_image: str
    reference_image_b: Optional[str] = None
    is_removal: bool = False
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    grounding_px: int = DEFAULT_GROUNDING_PX
    max_pixels: int = DEFAULT_MAX_PIXELS
    seed: int = -1


class ImageResponse(BaseModel):
    image_b64: str


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    model = AikoKrea2ImageGen()

    @web_app.post("/generate", response_model=ImageResponse)
    async def generate(req: GenerateRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        image_b64 = await model.generate.remote.aio(**req.model_dump())
        return ImageResponse(image_b64=image_b64)

    @web_app.post("/edit", response_model=ImageResponse)
    async def edit(req: EditRequest):
        if not req.instruction.strip():
            raise HTTPException(status_code=400, detail="instruction is required")
        if not req.reference_image:
            raise HTTPException(status_code=400, detail="reference_image is required")
        image_b64 = await model.identity_edit.remote.aio(**req.model_dump())
        return ImageResponse(image_b64=image_b64)

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "models": ["Krea-2-Raw", "Krea-2-Turbo"],
            "lora": "krea2-identity-edit-v1.1",
            "gpu": "A10G",
            "quantization": "nf4",
            "attn_implementation": "sdpa",
            "defaults": {
                "max_pixels": DEFAULT_MAX_PIXELS,
                "grounding_px": DEFAULT_GROUNDING_PX,
            },
        }

    return web_app