from __future__ import annotations

import requests

from social.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_linkedin",
        description="Post text + optional image URL to LinkedIn feed (requires OAuth access token)",
    )
    def post_linkedin(text: str, image_url: str | None = None) -> dict:
        access_token = env("LINKEDIN_ACCESS_TOKEN")
        author_id = env("LINKEDIN_AUTHOR_ID")

        if not access_token or not author_id:
            return {"ok": False, "provider": "linkedin", "error": "LINKEDIN_ACCESS_TOKEN or LINKEDIN_AUTHOR_ID not set"}

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        try:
            if image_url:
                payload = {
                    "author": f"urn:li:person:{author_id}",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {"text": text},
                            "shareMediaCategory": "IMAGE",
                            "media": [{"status": "READY", "media": image_url}],
                        }
                    },
                    "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
                }
            else:
                payload = {
                    "author": f"urn:li:person:{author_id}",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {"text": text},
                            "shareMediaCategory": "NONE",
                        }
                    },
                    "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
                }

            resp = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                headers=headers, json=payload, timeout=30,
            )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "provider": "linkedin", "status_code": resp.status_code, "response": resp.text[:1000]}
        except Exception as e:
            return {"ok": False, "provider": "linkedin", "error": str(e)}
