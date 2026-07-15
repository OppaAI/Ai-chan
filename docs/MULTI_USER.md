# Multi-user testing plan

There may be simple `$5 Patreon tester` isolation model for Aiko just to cover the cost of the front-end/back-end servers.
OAuth is the source of identity; `persona/identity.md` remains Aiko's identity only and is no longer a user identity source.

## Current implementation

- OAuth sessions store a provider-scoped runtime id, such as `github_123456` or `patreon_987654321`, to avoid collisions between providers.
- User-private state defaults to `~/.aiko/<user_id>/`:
  - `profile/user.md` for the user's editable bio/profile.
  - `memory.db` for that user's sqlite-vec memory store.
  - `monthly_consolidation_state.jsonl` for that user's monthly consolidation state.
  - `schedule.json` for that user's scheduled jobs/reminders (agentic scheduling).
  - `workspace/` for that user's notes and tool artifacts; this can later be redirected to a mounted/synced Google Drive workspace via `WORKSPACE_ROOT`.
  - `skills/` for that user's skill workflow state (future per-user skill customization).
- Existing env overrides still work for local/owner operation; leave YAML path overrides blank for per-user defaults:
  - `USER_ID` or `AIKO_USER_ID`
  - `SQLITE_MEMORY_PATH`
  - `USER_PROFILE_PATH`
  - `MONTHLY_CONSOLIDATION_STATE_PATH`
  - `SCHEDULE_PATH`
  - `WORKSPACE_ROOT`
  - `USER_STATE_ROOT` (canonical), plus compatibility aliases `AIKO_USER_STATE_ROOT` and `USER_SPACE_ROOT`

## sqlite-vec per user

One sqlite-vec database per user is fine for this tier. It is usually simpler and safer than a shared database because accidental unfiltered reads cannot cross user boundaries. The existing `user_id` column remains useful as a second safety belt and for migration/debugging.

## Encryption

Do **not** derive an encryption key from only the OAuth user id. User ids are not secret. If database encryption is added, derive keys from a server-side secret plus the provider/user id, for example with Argon2id or HKDF:

- secret input: `AIKO_DATA_KEY_SECRET` from the server environment or a KMS
- public context: `provider:user_id`
- output: per-user encryption key

`sqlite-vec` itself does not provide encryption with a normal SQLite `PRAGMA`. Aiko now has an optional SQLCipher hook for the memory DB: set `AIKO_SQLITE_ENCRYPTION=1`, install a SQLCipher-capable Python driver such as `pysqlcipher3`, and set `AIKO_DATA_KEY_SECRET` in `.env`, Modal secrets, or another secret manager. Aiko derives a per-user SQLCipher raw key from the server secret plus provider-scoped user id. This keeps latency low because encryption happens at SQLite page I/O, not per vector operation. If SQLCipher is not available in the deployment image, keep using OS/disk encryption, strict filesystem permissions, and per-user files/directories until the image is upgraded.

## Workspace encryption

For a simple online tester tier, prefer encrypting the storage layer instead of encrypting individual files in application code:

1. Put `USER_STATE_ROOT` on an encrypted persistent volume.
2. Restrict permissions to the service account.
3. Keep each user in a separate directory.
4. Avoid writing secrets into workspace files.

If the backend runs in Modal, the workspace is server-side unless you explicitly build a client upload/download feature. A browser app cannot silently read arbitrary files from a user's PC; users must upload files through a file picker, and Aiko can only access files that were uploaded to the backend workspace. To let a user take their workspace home, add an authenticated export endpoint that zips only `~/.aiko/<user_id>/workspace/` and streams it to the browser.

Per-file encryption can come later, but it complicates search, previews, scheduled jobs, and tool access because every tool must decrypt/re-encrypt correctly.

## Google Drive workspace direction

No agentic-tool changes are required just to move Aiko's writable workspace later. The existing workspace tools already go through `WORKSPACE_ROOT`. When Google Drive/Gmail access is added, initialize or mount/sync the Drive-backed workspace during boot/warmup, then point `WORKSPACE_ROOT` at that mounted directory. Keep private runtime state such as `memory.db`, `schedule.json`, and `monthly_consolidation_state.jsonl` under `~/.aiko/<user_id>/` rather than in Drive unless you explicitly want those files synced.

## Other per-user state to keep isolated

- OAuth sessions and terms acceptance records.
- Memory DB and daily reflection facts.
- Monthly consolidation state.
- Schedule/reminder state, workspace files, generated reports, and photo inboxes.
- User profile/bio markdown.
- Agentic skill workflow state and per-user skill customizations (future).
- Any future upload/cache directory that can contain private user content.

Global config YAML should stay owner-controlled for the `$5` tier. User-tunable settings can be added later for higher tiers by layering a per-user settings file over the global config.

## Hugging Face + Modal deployment notes

You can move model weights/assets to Hugging Face and point inference endpoints to Modal with modest code changes if the Modal services expose OpenAI-compatible endpoints already used by Aiko (`LLM_BASE_URL`, `EMBED_BASE_URL`, model names, and timeouts). The bigger online-deployment work is not model URLs; it is durable private storage, secrets management, OAuth callback configuration, process concurrency, rate limits, and backups.


## Low-latency DB encryption switch

Generate `AIKO_DATA_KEY_SECRET` once from a cryptographically secure random source and store it only in deployment secrets, not in git. Example:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Then set:

```bash
AIKO_SQLITE_ENCRYPTION=1
AIKO_DATA_KEY_SECRET=<the generated random secret>
```

With encryption enabled, Aiko opens `memory.db` through SQLCipher and validates the key during boot. Existing plaintext databases need a one-time migration/export into an encrypted database; do not simply flip the switch on an existing plaintext file.
