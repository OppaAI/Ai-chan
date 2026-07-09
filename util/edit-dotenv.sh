#!/usr/bin/env bash
# Decrypts .env.age to tmpfs, opens it in nvim, re-encrypts on save, and
# shreds the plaintext copy no matter how the script exits.
#
# Usage:
#   ./edit-dotenv.sh                       # uses $ENV_AGE_PATH, or ~/.aiko/.env.age
#   ./edit-dotenv.sh /path/to/.env.age     # overrides both of the above
#
# Env overrides:
#   AGE_KEY       path to the age identity (private key)   default: ~/.aiko/age-key.txt
#   AGE_KEY_PUB   path to the age recipient (public key)   default: ${AGE_KEY}.pub
#   ENV_AGE_PATH  path to the encrypted secrets file        default: ~/.aiko/.env.age

set -euo pipefail

ENC="${1:-${ENV_AGE_PATH:-$HOME/.aiko/.env.age}}"
KEY="${AGE_KEY:-$HOME/.aiko/age-key.txt}"
PUB="${AGE_KEY_PUB:-${KEY}.pub}"

if [ ! -f "$KEY" ]; then
    echo "age identity not found: $KEY" >&2
    exit 1
fi
if [ ! -f "$PUB" ]; then
    echo "age recipient not found: $PUB" >&2
    exit 1
fi

TMP="/dev/shm/aiko_env.$$"
SWAPDIR="/dev/shm/aiko_env_swap.$$"
mkdir -p "$SWAPDIR"

cleanup() {
    shred -u "$TMP" 2>/dev/null || rm -f "$TMP"
    rm -rf "$SWAPDIR"
}
trap cleanup EXIT INT TERM

if [ -f "$ENC" ]; then
    age -d -i "$KEY" -o "$TMP" "$ENC"
else
    : > "$TMP"
fi
chmod 600 "$TMP"

# -n disables swapfile entirely; directory is set anyway as a second layer
# in case a plugin forces one on.
nvim -n -c "set directory=${SWAPDIR}" "$TMP"

age -R "$PUB" -o "${ENC}.new" "$TMP"
mv "${ENC}.new" "$ENC"
echo "re-encrypted -> $ENC"