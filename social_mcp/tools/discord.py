from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

from social_mcp.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_discord",
        description="Send a message + optional image to a Discord channel via webhook or bot token",
    )
    def post_discord(text: str, image_path: str | None = None) -> dict:
        webhook_url = env("DISCORD_POST_WEBHOOK_URL")
        bot_token = env("DISCORD_BOT_TOKEN")
        channel_id = env("DISCORD_POST_CHANNEL_ID")

        if webhook_url:
            try:
                files = None
                data = {"content": text}
                if image_path:
                    p = Path(image_path)
                    if p.exists():
                        files = {"file": (p.name, open(p, "rb"))}
                resp = requests.post(webhook_url, data=data, files=files, timeout=30)
                ok = 200 <= resp.status_code < 300
                return {"ok": ok, "provider": "discord", "method": "webhook", "status_code": resp.status_code, "response": resp.text[:500]}
            except Exception as e:
                return {"ok": False, "provider": "discord", "method": "webhook", "error": str(e)}

        if bot_token and channel_id:
            try:
                api_base = env("DISCORD_API_BASE", "https://discord.com/api/v10").rstrip("/")
                headers = {"Authorization": f"Bot {bot_token}", "Content-Type": "application/json"}

                payload = {"content": text}
                files = None
                if image_path:
                    p = Path(image_path)
                    if p.exists():
                        mime = mimetypes.guess_type(str(p))[0] or "image/png"
                        with open(p, "rb") as f:
                            b64_data = base64.b64encode(f.read()).decode()
                        payload["attachments"] = [{"id": "0", "filename": p.name}]
                        payload["file"] = b64_data
                        headers.pop("Content-Type", None)
                        resp = requests.post(
                            f"{api_base}/channels/{channel_id}/messages",
                            headers={"Authorization": f"Bot {bot_token}"},
                            data={"payload_json": str(payload)},
                            files={"0": (p.name, open(p, "rb"), mime)},
                            timeout=30,
                        )
                    else:
                        resp = requests.post(
                            f"{api_base}/channels/{channel_id}/messages",
                            headers=headers, json=payload, timeout=30,
                        )
                else:
                    resp = requests.post(
                        f"{api_base}/channels/{channel_id}/messages",
                        headers=headers, json=payload, timeout=30,
                    )
                ok = 200 <= resp.status_code < 300
                return {"ok": ok, "provider": "discord", "method": "bot", "status_code": resp.status_code, "response": resp.text[:500]}
            except Exception as e:
                return {"ok": False, "provider": "discord", "method": "bot", "error": str(e)}

        return {"ok": False, "provider": "discord", "error": "No DISCORD_POST_WEBHOOK_URL or DISCORD_BOT_TOKEN+DISCORD_POST_CHANNEL_ID configured"}
