"""
system/tls.py
Resilient TLS CA-bundle resolution for outbound HTTP calls.

The runtime checkout lives on removable-media-backed storage; when that
filesystem hiccups mid-flight, the certifi bundle inside .venv can transiently
stop resolving and every requests call dies with:

    OSError: Could not find a suitable TLS CA certificate bundle, invalid path: ...

resolve_ca_bundle() returns the first bundle path that verifiably exists right
now (env override -> certifi -> OS trust store), and helpers let adapters and
direct requests.retry once against a healed bundle instead of failing the whole
job. Verification is never disabled — only re-pointed at a store that exists.
"""
from __future__ import annotations

import os

# Common OS trust-store locations (root filesystem, independent of any venv).
_SYSTEM_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",    # Fedora/RHEL
    "/etc/ssl/ca-bundle.pem",              # openSUSE
    "/usr/local/etc/ssl/cert.pem",         # FreeBSD/macOS ports
)

_ENV_VARS = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def _exists(path: str | None) -> bool:
    if not path:
        return False
    try:
        return os.path.exists(path)
    except OSError:
        return False


def resolve_ca_bundle(exclude: tuple[str, ...] | list[str] = ()) -> str | None:
    """Return an existing CA bundle path, or None for OpenSSL defaults.

    Order: explicit REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE env override (honored
    only while it exists), then the certifi wheel bundled in the active venv,
    then common OS trust stores. Paths in `exclude` are skipped — pass the
    path that just failed so a transient error on one candidate falls through
    to the next instead of repeating the failure.
    """
    excluded = {p for p in (exclude or ()) if p}
    candidates: list[str] = []
    for var in _ENV_VARS:
        value = os.getenv(var, "").strip()
        if value:
            candidates.append(value)
            break
    try:
        import certifi

        candidates.append(certifi.where())
    except Exception:
        pass
    candidates.extend(_SYSTEM_BUNDLES)
    for candidate in candidates:
        if not candidate or candidate in excluded:
            continue
        if _exists(candidate):
            return candidate
    return None


def is_ca_bundle_error(exc: BaseException) -> bool:
    """True for requests' missing-CA-bundle OSError family."""
    return isinstance(exc, OSError) and "CA certificate bundle" in str(exc)


def heal_verify(verify, exclude_current: bool = True):
    """Heal a failed `verify` setting into a working CA bundle path.

    Returns the replacement bundle path, or None when nothing valid can be
    resolved (caller should re-raise the original error).
    """
    failed = (verify,) if exclude_current and isinstance(verify, str) else ()
    return resolve_ca_bundle(exclude=failed)
