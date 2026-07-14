"""
CLI GitHub OAuth via Device Authorization Flow.

Reuses GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET from .env (same creds as
webui/auth.py) and the same allowlist env vars.  No local server needed
— the device flow shows a code the user enters at github.com/login/device.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from system.log import get_logger
from system.userspace import normalize_user_id

log = get_logger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

GITHUB_DEVICE_URL   = "https://github.com/login/device/code"
GITHUB_TOKEN_URL    = "https://github.com/login/oauth/access_token"
GITHUB_API_USER     = "https://api.github.com/user"
GITHUB_API_REPO     = "https://api.github.com/repos/{repo}/contributors"

TOKEN_FILE = Path.home() / ".aiko" / "auth_token.json"
SCOPES     = "read:user"

# How long (seconds) the device flow waits for the user to complete auth
# before giving up.  GitHub defaults to 15 min, 5 min is plenty for CLI.
_DEVICE_TIMEOUT = 300
_POLL_INTERVAL  = 5


# ── helpers ────────────────────────────────────────────────────────────────────


def _env_list(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


# ── public API ─────────────────────────────────────────────────────────────────


class CliAuth:
    """Thin wrapper around GitHub's OAuth device flow for the CLI."""

    def __init__(self) -> None:
        self.client_id       = os.getenv("GITHUB_CLIENT_ID", "")
        self.client_secret   = os.getenv("GITHUB_CLIENT_SECRET", "")
        self.allowed_users   = _env_list("ALLOWED_GITHUB_USERS")
        self.contrib_enabled = os.getenv("CONTRIBUTORS_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.repo            = os.getenv("GITHUB_REPO", "")

    # ── status ─────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Do we have a GitHub OAuth app configured in the environment?"""
        return bool(self.client_id)

    def get_stored_session(self) -> dict | None:
        """Return the persisted auth token + user info, or None."""
        if not TOKEN_FILE.exists():
            return None
        try:
            with TOKEN_FILE.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            TOKEN_FILE.unlink(missing_ok=True)
            return None

    def is_authenticated(self) -> bool:
        """Is a stored session present (and not obviously expired)?"""
        session = self.get_stored_session()
        if not session:
            return False
        # Crude expiry check: GitHub PATs typically last years, but the
        # OAuth device-flow token is an implicit grant token whose lifetime
        # is unspecified.  We verify it on use; if it fails we re-login.
        return bool(session.get("access_token"))

    def get_user_id(self) -> str:
        """Provider-scoped runtime id (e.g. 'github_123456') for filesystem paths."""
        session = self.get_stored_session()
        if session:
            user = session.get("user", {})
            uid = user.get("id")
            if uid:
                return normalize_user_id("github", uid)
        return "cli-user"

    def get_display_name(self) -> str:
        """GitHub login for display/prompt, not filesystem paths."""
        session = self.get_stored_session()
        if session:
            return session.get("user", {}).get("login", "cli-user")
        return "cli-user"

    def get_user_display(self) -> str:
        """Human-readable user identity."""
        session = self.get_stored_session()
        if not session:
            return "not authenticated"
        user = session.get("user", {})
        name = user.get("name") or user.get("login", "?")
        login = user.get("login", "?")
        return f"{name} ({login})" if name != login else login

    # ── login ──────────────────────────────────────────────────────────────

    def login(self) -> bool:
        """Run the GitHub device-authorisation flow in the terminal.

        Returns True on success (token stored, user authorised), False on
        any failure (timeout, network error, user not on allowlist, etc.).
        """
        if not self.is_configured():
            print("  ✗ GitHub OAuth not configured.  Set GITHUB_CLIENT_ID "
                  "and GITHUB_CLIENT_SECRET in your .env.")
            return False

        # 1 ── request a device code ────────────────────────────────────
        try:
            device_data = {"client_id": self.client_id, "scope": SCOPES}
            if self.client_secret:
                device_data["client_secret"] = self.client_secret
            resp = requests.post(
                GITHUB_DEVICE_URL,
                data=device_data,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            print(f"  ✗ Failed to contact GitHub: {exc}")
            log.warning("device-code request failed", exc_info=True)
            return False

        device_code      = body["device_code"]
        user_code        = body["user_code"]
        verification_uri = body.get("verification_uri",
                                    "https://github.com/login/device")
        interval         = body.get("interval", _POLL_INTERVAL)

        print()
        print(f"  → Open  {verification_uri}")
        print(f"  → Code  {user_code}")
        print("  Waiting for browser authorisation...")
        print("  (Ctrl-C to cancel)")

        # 2 ── poll for the access token ────────────────────────────────
        deadline = time.monotonic() + _DEVICE_TIMEOUT
        poll_interval = interval

        try:
            while time.monotonic() < deadline:
                time.sleep(poll_interval)

                poll_data = {
                    "client_id":     self.client_id,
                    "device_code":   device_code,
                    "grant_type":    "urn:ietf:params:oauth:"
                                     "grant-type:device_code",
                }
                if self.client_secret:
                    poll_data["client_secret"] = self.client_secret
                resp = requests.post(
                    GITHUB_TOKEN_URL,
                    data=poll_data,
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                token_body = resp.json()

                error = token_body.get("error")
                if error is None:
                    access_token = token_body.get("access_token")
                    if access_token:
                        return self._finalise(access_token)
                    print("  ✗ Unexpected response from GitHub.")
                    return False

                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    poll_interval += 5
                    continue
                if error in ("expired_token", "access_denied"):
                    print(f"  ✗ {error.replace('_', ' ').title()}.")
                    return False
                # Unknown error
                print(f"  ✗ GitHub returned: {error}")
                return False
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return False

        print("  ✗ Timed out waiting for authorisation.")
        return False

    def _finalise(self, access_token: str) -> bool:
        """Exchange the access token for a user profile, check allowlist,
        and persist the session locally."""
        # ── fetch user profile ──────────────────────────────────────────
        try:
            resp = requests.get(
                GITHUB_API_USER,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30,
            )
            resp.raise_for_status()
            user = resp.json()
        except requests.RequestException as exc:
            print(f"  ✗ Failed to fetch GitHub profile: {exc}")
            return False

        login = user.get("login", "")
        if not login:
            print("  ✗ Could not determine GitHub login.")
            return False

        # ── authorisation check ─────────────────────────────────────────
        if not self._is_authorised(login):
            print(f"  ✗ User '{login}' is not authorised for this instance.")
            log.info("CLI auth rejected user=%s", login)
            return False

        # ── persist ─────────────────────────────────────────────────────
        auth_data = {
            "access_token": access_token,
            "user":         user,
            "created_at":   time.time(),
        }
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(auth_data, indent=2))
        TOKEN_FILE.chmod(0o600)

        display = user.get("name") or login
        print(f"  ✓ Authenticated as {display} ({login})")
        log.info("CLI auth success user=%s provider=github", login)
        return True

    def _is_authorised(self, login: str) -> bool:
        """Check whether a GitHub login is permitted to use this instance."""
        login_lower = login.lower()

        # 1. Explicit allowlist (if non-empty, must be on it)
        if self.allowed_users:
            return login_lower in self.allowed_users

        # 2. Repo-contributor gate (requires GITHUB_REPO to be set)
        if self.contrib_enabled and self.repo:
            try:
                url = GITHUB_API_REPO.format(repo=self.repo)
                resp = requests.get(
                    url,
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    for entry in resp.json():
                        if isinstance(entry, dict) and entry.get("login", "").lower() == login_lower:
                            log.info("CLI auth contributor match user=%s repo=%s",
                                     login, self.repo)
                            return True
                log.warning("CLI auth contributor check failed "
                            "repo=%s status=%d", self.repo, resp.status_code)
            except requests.RequestException as exc:
                log.warning("CLI auth contributor check error repo=%s: %s",
                            self.repo, exc)

        # 3. No allowlist and no contributor gate → open (admin mode)
        return not self.allowed_users and not (self.contrib_enabled and self.repo)

    # ── logout ─────────────────────────────────────────────────────────────

    def logout(self) -> None:
        """Remove the stored auth token."""
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
            print("  ✓ Logged out.")
        else:
            print("  Not logged in.")
