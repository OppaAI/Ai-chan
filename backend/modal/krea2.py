"""
krea2-img2img.py — Krea 2 Raw image generation endpoint for Aiko, with SDEdit-style
reference-image conditioning (img2img via partial noise injection).
Modal app: oppa-ai-org--aiko-imagegen-krea2

IMPORTANT — read before relying on this:
Krea2Pipeline (as of the diffusers source checked 2026-07-03) is text-to-image ONLY.
There is no `image` / `images` kwarg like Flux2KleinPipeline has — the Krea 2 DiT's
token sequence is [text_tokens, noisy_image_patches] with no reference-latent slot
(confirmed by the ComfyUI-Krea2TextEncoder project notes: "there is no slot for a
reference latent ... it can't be done from a node alone").

What THIS script does instead: classic SDEdit img2img. We VAE-encode your reference
image, partially re-noise it to a chosen `strength`, and only run the LAST
`strength * num_inference_steps` steps of the normal denoising loop starting from
that partially-noised latent instead of from pure Gaussian noise. This works because
SDEdit only touches the *input* to the existing noisy_image_patches slot — it doesn't
need a new conditioning pathway, so it's fully compatible with Krea 2's architecture.
This gives you style/composition-guided generation, NOT pixel-faithful editing and NOT
true multi-reference/IP-adapter-style conditioning.

strength guide (same convention as diffusers img2img pipelines):
  strength=1.0  -> ignores the reference image entirely (pure text-to-image)
  strength=0.6-0.8 -> loose composition/style guidance, prompt still drives a lot
  strength=0.3-0.5 -> strong resemblance to the reference, prompt nudges details
  strength closer to 0.0 -> barely changes the reference at all

Use Krea 2 RAW for this, not Turbo — Turbo's 8-step TDM-distilled trajectory is
trained to run start-to-finish from pure noise with CFG disabled, so injecting
partially-noised latents mid-trajectory puts it in an out-of-distribution state.
Raw's 52-step schedule with real CFG gives you both the step resolution to pick a
sensible strength and a CFG knob to balance prompt-faithfulness vs. reference-faithfulness.

Requires Modal secrets:
  - huggingface-secret (HF_TOKEN)
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
        # Krea2Pipeline is very recent — install diffusers from source.
        "git+https://github.com/huggingface/diffusers.git",
        # Qwen3-VL text encoder support is also recent — install transformers from source
        # to be safe. If you've already pinned a working transformers version on AIVA/AuRoRA
        # for another project, swap this for that exact pin instead to avoid drift.
        "git+https://github.com/huggingface/transformers.git",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "Pillow",
        "numpy",
        "fastapi[standard]",
    )
    .env({"DIFFUSERS_NO_FLASH_ATTN": "1"})
)

volume = modal.Volume.from_name("aiko-imagegen-krea2-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
MODEL_ID = "krea/Krea-2-Raw"  # Raw, not Turbo — see docstring above

app = modal.App("aiko-imagegen-krea2", image=image)


# ---------------------------------------------------------------------------
# one-time weight downloader — run with: modal run krea2-img2img.py::download_weights
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: volume},
    timeout=3600,
)
def download_weights():
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.environ["HF_TOKEN"]
    local_path = f"{WEIGHTS_DIR}/krea2-raw"

    print("Downloading Krea 2 Raw weights...")
    snapshot_download(
        MODEL_ID,
        local_dir=local_path,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5"],
    )
    volume.commit()
    print("Done. Volume committed.")


# ---------------------------------------------------------------------------
# model class
# ---------------------------------------------------------------------------
@app.cls(
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_DIR: volume},
    timeout=180,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class AikoKrea2ImageGen:

    @modal.enter()
    def load(self):
        import os
        import torch
        from diffusers import Krea2Pipeline

        local_path = f"{WEIGHTS_DIR}/krea2-raw"
        shard_check = f"{local_path}/transformer"

        if not os.path.exists(local_path) or not os.path.exists(shard_check):
            from huggingface_hub import snapshot_download
            hf_token = os.environ["HF_TOKEN"]
            print("Weights missing or incomplete — downloading...")
            snapshot_download(
                MODEL_ID,
                local_dir=local_path,
                token=hf_token,
                ignore_patterns=["*.msgpack", "*.h5"],
            )
            volume.commit()
        else:
            print("Weights cached, loading from volume...")

        self.pipe = Krea2Pipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("Krea 2 Raw ready.")

    # -----------------------------------------------------------------
    # SDEdit img2img core — mirrors Krea2Pipeline.__call__ internals but
    # starts the denoising loop from a partially-noised reference latent
    # instead of pure Gaussian noise. See module docstring for caveats.
    # -----------------------------------------------------------------
    def _encode_reference_image(self, pil_image, height, width):
        import torch

        pipe = self.pipe
        # VaeImageProcessor.preprocess -> (B, C, H, W) in [-1, 1]
        image_tensor = pipe.image_processor.preprocess(pil_image, height=height, width=width)
        image_tensor = image_tensor.to(device="cuda", dtype=pipe.vae.dtype)
        # QwenImage VAE is a video-style VAE expecting (B, C, T, H, W); T=1 for a still image.
        image_tensor = image_tensor.unsqueeze(2)

        with torch.no_grad():
            latent_dist = pipe.vae.encode(image_tensor).latent_dist
            raw_latents = latent_dist.mode()  # deterministic encode, no extra sampling noise

        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(raw_latents.device, raw_latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(raw_latents.device, raw_latents.dtype)
        )
        # Inverse of the normalization applied at decode time in Krea2Pipeline.__call__:
        # decode does `latents = latents / latents_std + latents_mean`, so encode-side is:
        norm_latents = (raw_latents - latents_mean) * latents_std  # (B, z_dim, 1, latH, latW)
        norm_latents = norm_latents.squeeze(2)  # drop T -> (B, z_dim, latH, latW)

        batch_size, num_channels_latents, latent_height, latent_width = norm_latents.shape
        packed = pipe._pack_latents(norm_latents, batch_size, num_channels_latents, latent_height, latent_width)
        return packed

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 52,
        guidance_scale: float = 3.5,
        strength: float = 0.65,
        seed: int = -1,
        reference_image: str | None = None,  # single base64-encoded PNG/JPG string
        negative_prompt: str = "",
        max_sequence_length: int = 512,
    ) -> str:
        """Generate image, return base64-encoded PNG string.

        If reference_image is None, this is plain text-to-image (equivalent to
        calling self.pipe(prompt, ...) directly). If reference_image is provided,
        this runs SDEdit img2img starting from that image's noised latents.
        """
        import numpy as np
        import torch
        from PIL import Image

        pipe = self.pipe
        device = "cuda"

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)

        do_cfg = guidance_scale > 0

        # ---- 1. Encode prompts (reuses the pipeline's own method) ----
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = None, None
        if do_cfg:
            neg = negative_prompt or ""
            negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
                prompt=neg,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        # ---- 2. Prepare timesteps (mirrors Krea2Pipeline.__call__ step 5) ----
        sigmas = np.linspace(1.0, 1 / steps, steps)
        num_channels_latents = pipe.transformer.config.in_channels // (pipe.patch_size**2)

        if reference_image is not None:
            image_latents = self._encode_reference_image(
                Image.open(io.BytesIO(base64.b64decode(reference_image))).convert("RGB"),
                height,
                width,
            )
            image_seq_len = image_latents.shape[1]
        else:
            image_seq_len = (height // (pipe.vae_scale_factor * pipe.patch_size)) * (
                width // (pipe.vae_scale_factor * pipe.patch_size)
            )

        if pipe.config.is_distilled:
            mu = 1.15
        else:
            from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

            mu = calculate_shift(
                image_seq_len,
                pipe.scheduler.config.get("base_image_seq_len", 256),
                pipe.scheduler.config.get("max_image_seq_len", 6400),
                pipe.scheduler.config.get("base_shift", 0.5),
                pipe.scheduler.config.get("max_shift", 1.15),
            )

        pipe.scheduler.set_timesteps(steps, device=device, sigmas=sigmas, mu=mu)
        timesteps = pipe.scheduler.timesteps

        # ---- 3. Prepare latents ----
        grid_height = height // (pipe.vae_scale_factor * pipe.patch_size)
        grid_width = width // (pipe.vae_scale_factor * pipe.patch_size)
        position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)

        if reference_image is not None:
            strength = max(0.0, min(1.0, strength))
            init_timestep = max(int(steps * strength), 1)
            t_start = max(steps - init_timestep, 0)
            timesteps = timesteps[t_start:]
            pipe.scheduler.set_begin_index(t_start)

            noise = torch.randn(
                image_latents.shape, generator=generator, device=device, dtype=image_latents.dtype
            )
            latent_timestep = timesteps[:1].expand(image_latents.shape[0])
            latents = pipe.scheduler.scale_noise(image_latents, latent_timestep, noise)
        else:
            pipe.scheduler.set_begin_index(0)
            latents = pipe.prepare_latents(
                1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator
            )

        # ---- 4. Denoising loop (mirrors Krea2Pipeline.__call__ step 6) ----
        with torch.no_grad():
            for t in timesteps:
                timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)

                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]

                if do_cfg:
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 5. Decode (mirrors Krea2Pipeline.__call__ step 7) ----
        latents = pipe._unpack_latents(latents, height, width)
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        pil_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

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
    steps: int = 52  # 52 for Raw; drop to ~28 if you want faster/rougher previews
    guidance_scale: float = 3.5
    strength: float = 0.65  # only used when reference_image is set
    seed: int = -1
    reference_image: Optional[str] = None  # single base64 string


class GenerateResponse(BaseModel):
    image_b64: str
    prompt: str


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    model = AikoKrea2ImageGen()

    @web_app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        image_b64 = await model.generate.remote.aio(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            seed=req.seed,
            reference_image=req.reference_image,
        )
        return GenerateResponse(image_b64=image_b64, prompt=req.prompt)

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": "Krea-2-Raw"}

    return web_app