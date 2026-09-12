#!/usr/bin/env bash
# Syncs the single source of truth for the server URL (AIKO_PUBLIC_BASE_URL)
# into the Android apps' build config.
#
# Source precedence (mirrors system/config.py):
#   1. age-encrypted dotenv (user-managed via util/edit_dotenv.sh)
#   2. config/android_app.yaml AIKO_PUBLIC_BASE_URL
#      (keep it "" in YAML when .env.age holds the value — YAML wins)
#
# Writes aikoServerUrl=<url> into:
#   Aiko-Games/local.properties and Aiko-Lingo/local.properties
# Rebuild each APK afterwards to pick it up (baked into BuildConfig).
#
# Usage:
#   ./util/sync_app_server_url.sh              # sync both apps
#   ./util/sync_app_server_url.sh --print-only # just print the resolved URL
#
# Env overrides:
#   ENV_AGE_PATH  path to encrypted secrets   default: ~/.aiko/.env.age
#   AGE_KEY       path to age identity         default: ~/.aiko/age-key.txt
#   GAMES_DIR     Aiko-Games checkout          default: <repo>/../Aiko-Games
#   LINGO_DIR     Aiko-Lingo checkout          default: <repo>/../Aiko-Lingo

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENC="${ENV_AGE_PATH:-$HOME/.aiko/.env.age}"
KEY="${AGE_KEY:-$HOME/.aiko/age-key.txt}"
GAMES_DIR="${GAMES_DIR:-$(dirname "$REPO")/Aiko-Games}"
LINGO_DIR="${LINGO_DIR:-$(dirname "$REPO")/Aiko-Lingo}"

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

unquote() {
    local s="$1"
    if [[ ${#s} -ge 2 ]]; then
        if [[ "${s:0:1}" == '"' && "${s: -1}" == '"' ]] ||
            [[ "${s:0:1}" == "'" && "${s: -1}" == "'" ]]; then
            s="${s:1:-1}"
        fi
    fi
    printf '%s' "$s"
}

dotenv_value() { # $1=file $2=key -> value or ""
    local file="$1" key="$2" line val
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$file" 2>/dev/null | tail -n 1 || true)"
    [[ -z "$line" ]] && return 0
    val="${line#*=}"
    val="${val%%#*}" # strip trailing comment (values here are URLs, no # inside)
    unquote "$(trim "$val")"
}

resolve_url() {
    local url=""
    # 1. age-encrypted dotenv
    if [[ -f "$ENC" ]]; then
        if [[ ! -f "$KEY" ]]; then
            echo "error: $ENC exists but age identity not found: $KEY" >&2
            exit 1
        fi
        local tmp
        tmp="$(mktemp)"
        trap 'rm -f "$tmp"' RETURN
        age -d -i "$KEY" -o "$tmp" "$ENC"
        url="$(dotenv_value "$tmp" AIKO_PUBLIC_BASE_URL)"
    fi
    # 2. YAML fallback (only when .env.age did not define it)
    if [[ -z "$url" && -f "$REPO/config/android_app.yaml" ]]; then
        local line val
        line="$(grep -E "^AIKO_PUBLIC_BASE_URL:" "$REPO/config/android_app.yaml" | tail -n 1 || true)"
        if [[ -n "$line" ]]; then
            val="${line#*:}"
            val="${val%%#*}"
            url="$(unquote "$(trim "$val")")"
        fi
    fi
    if [[ -z "$url" ]]; then
        cat >&2 <<'EOF'
error: AIKO_PUBLIC_BASE_URL is not set anywhere.
Add it with one of:
  1. util/edit_dotenv.sh   # then add: AIKO_PUBLIC_BASE_URL=https://<tailnet>/
     (file: ~/.aiko/.env.age — preferred, encrypted)
  2. config/android_app.yaml -> AIKO_PUBLIC_BASE_URL: "https://<tailnet>/"
     (only when .env.age does NOT define it — YAML wins)
EOF
        exit 1
    fi
    printf '%s\n' "$url"
}

set_prop() { # $1=file $2=key $3=value
    local file="$1" key="$2" value="$3"
    if [[ -f "$file" ]] && grep -qE "^${key}=" "$file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >>"$file"
    fi
    echo "  wrote ${key}=${value} -> $file"
}

URL="$(resolve_url)"

if [[ "${1:-}" == "--print-only" ]]; then
    printf '%s\n' "$URL"
    exit 0
fi

echo "Resolved AIKO_PUBLIC_BASE_URL=$URL"
for dir in "$GAMES_DIR" "$LINGO_DIR"; do
    if [[ ! -d "$dir/app" ]]; then
        echo "  skip $dir (not found)" >&2
        continue
    fi
    set_prop "$dir/local.properties" "aikoServerUrl" "$URL"
done
echo "Done. Rebuild the APKs to bake in the new URL."
