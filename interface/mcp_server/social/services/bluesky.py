
from pathlib import Path

from social.services import env


def load_tools(mcp):
    @mcp.tool(
        name="post_bluesky",
        description="Post text + optional image to Bluesky",
    )
    def post_bluesky(text: str, image_path: str | None = None) -> dict:
        handle = env("BLUESKY_HANDLE")
        app_pass = env("BLUESKY_APP_PASS")

        if not handle or not app_pass:
            return {"ok": False, "provider": "bluesky", "error": "BLUESKY_HANDLE or BLUESKY_APP_PASS not set"}

        try:
            from atproto import Client, models
        except ImportError:
            return {"ok": False, "provider": "bluesky", "error": "atproto not installed — pip install atproto"}

        try:
            client = Client()
            client.login(handle, app_pass)

            if image_path:
                p = Path(image_path)
                if not p.exists():
                    return {"ok": False, "provider": "bluesky", "error": f"image not found: {image_path}"}
                with open(p, "rb") as f:
                    img_data = f.read()
                upload = client.upload_blob(img_data)
                embed = models.AppBskyEmbedImages.Main(
                    images=[models.AppBskyEmbedImages.Image(alt="", image=upload.blob)]
                )
                post = client.send_post(text=text, embed=embed)
            else:
                post = client.send_post(text=text)

            return {"ok": True, "provider": "bluesky", "uri": post.uri, "cid": post.cid}
        except Exception as e:
            return {"ok": False, "provider": "bluesky", "error": str(e)}
