from __future__ import annotations

from social.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_reddit",
        description="Post a text, link, or image post to Reddit",
    )
    def post_reddit(title: str, text: str = "", image_path: str | None = None, subreddit: str = "") -> dict:
        client_id = env("REDDIT_CLIENT_ID")
        client_secret = env("REDDIT_CLIENT_SECRET")
        username = env("REDDIT_USERNAME")
        password = env("REDDIT_PASSWORD")
        user_agent = env("REDDIT_USER_AGENT", "Aiko-chan v0.1")

        if not all([client_id, client_secret, username, password]):
            return {"ok": False, "provider": "reddit", "error": "Reddit credentials not configured"}

        try:
            import praw
        except ImportError:
            return {"ok": False, "provider": "reddit", "error": "praw not installed — pip install praw"}

        try:
            reddit = praw.Reddit(
                client_id=client_id, client_secret=client_secret,
                username=username, password=password, user_agent=user_agent,
            )
            target = reddit.subreddit(subreddit) if subreddit else reddit.user.me().subreddit

            if image_path:
                submission = target.submit_image(title, image_path)
                if text:
                    submission.reply(text)
            else:
                submission = target.submit(title, selftext=text or "")

            return {"ok": True, "provider": "reddit", "subreddit": str(target), "submission_id": submission.id, "url": submission.url}
        except Exception as e:
            return {"ok": False, "provider": "reddit", "error": str(e)}
