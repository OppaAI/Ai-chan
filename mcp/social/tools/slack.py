from __future__ import annotations

from pathlib import Path

import requests

from social.tools.base import env, int_env


def load_tools(mcp):
    @mcp.tool(
        name="post_slack",
        description="Send a message + optional image to a Slack channel via webhook or bot token",
    )
    def post_slack(text: str, channel: str | None = None, image_path: str | None = None) -> dict:
        webhook_url = env("SLACK_POST_WEBHOOK_URL")
        bot_token = env("SLACK_BOT_TOKEN")
        default_channel = env("SLACK_POST_CHANNEL")
        target_channel = channel or default_channel

        if webhook_url:
            try:
                payload = {"text": text}
                if image_path:
                    p = Path(image_path)
                    if p.exists():
                        import base64, mimetypes
                        mime = mimetypes.guess_type(str(p))[0] or "image/png"
                        with open(p, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        payload["attachments"] = [{"fallback": text, "image_url": f"data:{mime};base64,{b64}"}]
                resp = requests.post(webhook_url, json=payload, timeout=30)
                ok = 200 <= resp.status_code < 300
                return {"ok": ok, "provider": "slack", "method": "webhook", "status_code": resp.status_code, "response": resp.text[:500]}
            except Exception as e:
                return {"ok": False, "provider": "slack", "method": "webhook", "error": str(e)}

        if bot_token:
            try:
                api_base = env("SLACK_API_BASE", "https://slack.com/api")
                headers = {"Authorization": f"Bearer {bot_token}", "Content-Type": "application/json"}
                if not target_channel:
                    return {"ok": False, "provider": "slack", "error": "No channel specified and SLACK_POST_CHANNEL not set"}
                payload = {"channel": target_channel, "text": text}
                if image_path:
                    import mimetypes, base64
                    p = Path(image_path)
                    if p.exists():
                        mime = mimetypes.guess_type(str(p))[0] or "image/png"
                        upload = requests.post(
                            f"{api_base}/files.upload",
                            headers={"Authorization": f"Bearer {bot_token}"},
                            data={"channels": target_channel, "initial_comment": text},
                            files={"file": (p.name, open(p, "rb"), mime)},
                            timeout=30,
                        )
                        return {"ok": upload.status_code == 200, "provider": "slack", "method": "bot_upload", "status_code": upload.status_code, "response": upload.text[:500]}
                resp = requests.post(f"{api_base}/chat.postMessage", headers=headers, json=payload, timeout=30)
                ok = 200 <= resp.status_code < 300
                return {"ok": ok, "provider": "slack", "method": "bot", "status_code": resp.status_code, "response": resp.text[:500]}
            except Exception as e:
                return {"ok": False, "provider": "slack", "error": str(e)}

        return {"ok": False, "provider": "slack", "error": "No SLACK_POST_WEBHOOK_URL or SLACK_BOT_TOKEN configured"}
