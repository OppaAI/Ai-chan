"""
flux-klein.py — FLUX.2 [klein] 9B image generation endpoint for Aiko
Modal app: oppa-ai-org--aiko-imagegen

Supports:
  - Text-to-image (no reference_images)
  - Multi-reference image-to-image (pass 1-2 base64 PNG/JPG strings)

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
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "Pillow",
        "fastapi[standard]",
    )
    .env({"DIFFUSERS_NO_FLASH_ATTN": "1"})
)

volume = modal.Volume.from_name("aiko-imagegen-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

app = modal.App("aiko-imagegen", image=image)

# ---------------------------------------------------------------------------
# one-time weight downloader — run with: modal run flux-klein.py::download_weights
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
    local_path = f"{WEIGHTS_DIR}/flux2-klein-9b"

    print("Downloading FLUX.2 klein 9B weights...")
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
    timeout=120,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class AikoImageGen:

    @modal.enter()
    def load(self):
        import os
        import torch
        from diffusers import Flux2KleinPipeline

        local_path = f"{WEIGHTS_DIR}/flux2-klein-9b"
        shard_check = f"{local_path}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"

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

        self.pipe = Flux2KleinPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("FLUX.2 klein 9B ready.")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int = -1,
        reference_images: list[str] | None = None,  # base64-encoded PNG/JPG strings
    ) -> str:
        """Generate image, return base64-encoded PNG string."""
        import torch
        from PIL import Image

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # decode reference images if provided
        ref_pil_images = []
        if reference_images:
            for b64 in reference_images:
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ref_pil_images.append(img)

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if ref_pil_images:
            # FLUX.2 klein i2i: pass as `image` (single) or `images` (multi-reference)
            if len(ref_pil_images) == 1:
                kwargs["image"] = ref_pil_images[0]
            else:
                kwargs["image"] = ref_pil_images  # multi-reference

        result = self.pipe(**kwargs)
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
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
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance_scale: float = 1.0
    seed: int = -1
    reference_images: Optional[list[str]] = None  # base64 strings


class GenerateResponse(BaseModel):
    image_b64: str
    prompt: str


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    model = AikoImageGen()

    @web_app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        image_b64 = await model.generate.remote.aio(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            reference_images=req.reference_images,
        )
        return GenerateResponse(image_b64=image_b64, prompt=req.prompt)

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": "FLUX.2-klein-9B"}

    return web_app
    