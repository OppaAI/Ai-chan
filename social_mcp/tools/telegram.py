from __future__ import annotations

from pathlib import Path

import requests

from social_mcp.tools.base import env, int_env


def load_tools(mcp):
    @mcp.tool(
        name="post_telegram",
        description="Send text + optional image to a Telegram chat via bot",
    )
    def post_telegram(text: str, image_path: str | None = None, chat_id: str | None = None) -> dict:
        bot_token = env("TELEGRAM_BOT_TOKEN")
        default_chat_id = env("TELEGRAM_POST_CHAT_ID")
        cid = chat_id or default_chat_id

        if not bot_token or not cid:
            return {"ok": False, "provider": "telegram", "error": "TELEGRAM_BOT_TOKEN or TELEGRAM_POST_CHAT_ID not set"}

        api_base = env("TELEGRAM_API_BASE", f"https://api.telegram.org/bot{bot_token}")
        timeout = int_env("TELEGRAM_POST_TIMEOUT", 30)

        try:
            if image_path:
                p = Path(image_path)
                if not p.exists():
                    return {"ok": False, "provider": "telegram", "error": f"image not found: {image_path}"}
                with open(p, "rb") as f:
                    resp = requests.post(
                        f"{api_base}/sendPhoto",
                        data={"chat_id": cid, "caption": text},
                        files={"photo": f},
                        timeout=timeout,
                    )
            else:
                resp = requests.post(
                    f"{api_base}/sendMessage",
                    json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                    timeout=timeout,
                )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "provider": "telegram", "status_code": resp.status_code, "response": resp.text[:500]}
        except Exception as e:
            return {"ok": False, "provider": "telegram", "error": str(e)}
