"""
tests/unit/test_config.py
Tests for the config bootstrap module (assumed at system/config.py --
adjust the import below if it lives elsewhere; the module docstring's
`root = Path(__file__).resolve().parent.parent` implies it sits one level
under the repo root, same as system/userspace.py and system/secure.py).

This module has real branching precedence logic (env > YAML > .env.age,
except under override=True where processing ORDER decides the winner),
a hand-rolled YAML fallback parser, and a subprocess call out to the `age`
binary -- all of which deserve direct coverage, not just a smoke test.

Layers covered:
  1. Pure helpers      — _strip_comment, _simple_yaml_load, _stringify, _flatten
  2. _decrypt_env       — subprocess/age interaction, fully mocked (no real
                          age binary or key required)
  3. load_config        — full precedence integration, using tmp_path
                          scaffolding and a monkeypatched module __file__ so
                          "root" points at a throwaway directory instead of
                          the real repo
"""
from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from system import config as config_module
from system.config import (
    _decrypt_env,
    _flatten,
    _load_yaml_mapping,
    _simple_yaml_load,
    _stringify,
    _strip_comment,
    load_config,
)


@pytest.fixture(autouse=True)
def reset_loaded_flag(monkeypatch):
    """_LOADED is a module-level singleton flag -- reset it before every
    test so load_config() actually re-runs instead of no-op'ing due to a
    previous test's call."""
    monkeypatch.setattr(config_module, "_LOADED", False)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "USER_STATE_ROOT", "AGE_KEY", "ENV_AGE_PATH",
        "SOME_KEY", "NESTED_A_B", "LIST_KEY", "OVERRIDE_ME", "SECRET_ONLY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStripComment:
    def test_strips_trailing_comment(self):
        assert _strip_comment("key: value # a comment").strip() == "key: value"

    def test_hash_inside_double_quotes_preserved(self):
        result = _strip_comment('key: "value # not a comment"')
        assert "# not a comment" in result

    def test_hash_inside_single_quotes_preserved(self):
        result = _strip_comment("key: 'value # not a comment'")
        assert "# not a comment" in result

    def test_no_comment_returns_unchanged(self):
        assert _strip_comment("key: value") == "key: value"


class TestSimpleYamlLoad:
    def test_basic_scalar_pairs(self):
        text = "foo: bar\nbaz: 42\n"
        result = _simple_yaml_load(text)
        assert result == {"foo": "bar", "baz": "42"}

    def test_strips_quotes_from_values(self):
        text = 'foo: "bar"\nbaz: \'qux\'\n'
        result = _simple_yaml_load(text)
        assert result == {"foo": "bar", "baz": "qux"}

    def test_blank_lines_and_comments_ignored(self):
        text = "\n# a comment line\nfoo: bar\n\n"
        result = _simple_yaml_load(text)
        assert result == {"foo": "bar"}

    def test_empty_scalar_stays_empty_string_if_no_list_follows(self):
        text = "foo:\nbar: baz\n"
        result = _simple_yaml_load(text)
        assert result["foo"] == ""
        assert result["bar"] == "baz"

    def test_ambiguous_empty_key_promoted_to_list_when_items_follow(self):
        text = "configs:\n  - a.yaml\n  - b.yaml\n"
        result = _simple_yaml_load(text)
        assert result["configs"] == ["a.yaml", "b.yaml"]

    def test_list_items_have_quotes_stripped(self):
        text = 'configs:\n  - "a.yaml"\n  - \'b.yaml\'\n'
        result = _simple_yaml_load(text)
        assert result["configs"] == ["a.yaml", "b.yaml"]

    def test_line_without_colon_is_ignored(self):
        text = "foo: bar\nnot a valid line without colon\nbaz: qux\n"
        result = _simple_yaml_load(text)
        assert result == {"foo": "bar", "baz": "qux"}


class TestStringify:
    def test_none_becomes_empty_string(self):
        assert _stringify(None) == ""

    def test_bool_true_becomes_one(self):
        assert _stringify(True) == "1"

    def test_bool_false_becomes_zero(self):
        assert _stringify(False) == "0"

    def test_list_becomes_json_array_of_strings(self):
        import json
        result = _stringify(["a", "b", 3])
        assert json.loads(result) == ["a", "b", "3"]

    def test_tuple_becomes_json_array(self):
        import json
        result = _stringify(("x", "y"))
        assert json.loads(result) == ["x", "y"]

    def test_plain_scalar_becomes_str(self):
        assert _stringify(42) == "42"
        assert _stringify("already a string") == "already a string"


class TestFlatten:
    def test_flat_dict_uppercased(self):
        result = _flatten({"foo": "bar"})
        assert result == {"FOO": "bar"}

    def test_nested_dict_joined_with_underscore(self):
        result = _flatten({"nested": {"a": {"b": 1}}})
        assert result == {"NESTED_A_B": 1}

    def test_mixed_flat_and_nested(self):
        result = _flatten({"top": "value", "group": {"child": "x"}})
        assert result == {"TOP": "value", "GROUP_CHILD": "x"}

    def test_empty_dict_returns_empty(self):
        assert _flatten({}) == {}


class TestLoadYamlMapping:
    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert _load_yaml_mapping(p) == {}

    def test_falls_back_to_simple_parser_when_pyyaml_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "yaml", None)
        p = tmp_path / "test.yaml"
        p.write_text("foo: bar\n")
        result = _load_yaml_mapping(p)
        assert result == {"foo": "bar"}

    def test_uses_real_yaml_when_available(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "test.yaml"
        p.write_text("foo: bar\nnum: 42\n")
        result = _load_yaml_mapping(p)
        assert result["foo"] == "bar"
        assert result["num"] == 42  # real PyYAML preserves int type, unlike the fallback


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — _decrypt_env: subprocess/age interaction, fully mocked
# ─────────────────────────────────────────────────────────────────────────────

class TestDecryptEnv:
    def test_returns_empty_dict_when_enc_file_missing(self, tmp_path):
        result = _decrypt_env(tmp_path / "missing.env.age", tmp_path / "key.txt")
        assert result == {}

    def test_raises_when_identity_file_missing(self, tmp_path):
        enc = tmp_path / "secrets.env.age"
        enc.write_text("dummy")  # content doesn't matter, identity check happens first
        with pytest.raises(FileNotFoundError, match="AGE_KEY"):
            _decrypt_env(enc, tmp_path / "nonexistent-key.txt")

    def test_raises_runtime_error_when_age_binary_missing(self, tmp_path, monkeypatch):
        enc = tmp_path / "secrets.env.age"
        enc.write_text("dummy")
        key = tmp_path / "key.txt"
        key.write_text("dummy-key")

        def _raise_not_found(*args, **kwargs):
            raise FileNotFoundError("age not found")

        monkeypatch.setattr(subprocess, "run", _raise_not_found)
        with pytest.raises(RuntimeError, match="age.*binary"):
            _decrypt_env(enc, key)

    def test_raises_runtime_error_on_decrypt_failure(self, tmp_path, monkeypatch):
        enc = tmp_path / "secrets.env.age"
        enc.write_text("dummy")
        key = tmp_path / "key.txt"
        key.write_text("dummy-key")

        def _raise_called_process_error(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd=["age"], stderr=b"bad key or corrupt file"
            )

        monkeypatch.setattr(subprocess, "run", _raise_called_process_error)
        with pytest.raises(RuntimeError, match="bad key or corrupt file"):
            _decrypt_env(enc, key)

    def test_successful_decrypt_parses_dotenv_stdout(self, tmp_path, monkeypatch):
        pytest.importorskip("dotenv")
        enc = tmp_path / "secrets.env.age"
        enc.write_text("dummy")
        key = tmp_path / "key.txt"
        key.write_text("dummy-key")

        fake_stdout = b"API_TOKEN=abc123\nSOME_URL=https://example.com\n"

        class _FakeCompletedProcess:
            stdout = fake_stdout
            stderr = b""

        def _fake_run(*args, **kwargs):
            return _FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = _decrypt_env(enc, key)
        assert result == {"API_TOKEN": "abc123", "SOME_URL": "https://example.com"}

    def test_never_writes_plaintext_to_disk(self, tmp_path, monkeypatch):
        """Regression guard for the documented guarantee: 'Plaintext is
        never written to disk'. Confirms no new files appear in tmp_path
        after a decrypt call beyond the enc/key files that were already
        there."""
        pytest.importorskip("dotenv")
        enc = tmp_path / "secrets.env.age"
        enc.write_text("dummy")
        key = tmp_path / "key.txt"
        key.write_text("dummy-key")
        before = set(tmp_path.iterdir())

        class _FakeCompletedProcess:
            stdout = b"SECRET=value\n"
            stderr = b""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
        _decrypt_env(enc, key)

        after = set(tmp_path.iterdir())
        assert after == before, "decrypt_env created unexpected file(s) on disk"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — load_config(): full precedence integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """
    Redirects the module's notion of "repo root" to a throwaway tmp_path
    by monkeypatching __file__, since `root = Path(__file__).resolve()
    .parent.parent` is computed fresh inside load_config() every call.
    Layout created: <tmp_path>/pkg/config.py (fake, unused on disk) so
    root resolves to <tmp_path>.
    """
    fake_module_path = tmp_path / "pkg" / "config.py"
    monkeypatch.setattr(config_module, "__file__", str(fake_module_path))
    return tmp_path


def _write_yaml(root: Path, name: str, content: str) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    p = config_dir / name
    p.write_text(content)
    return p


class TestLoadConfigYamlPrecedence:
    def test_yaml_sets_env_var_when_not_already_set(self, fake_root, monkeypatch):
        _write_yaml(fake_root, "a.yaml", "some_key: from_yaml\n")
        load_config()
        assert os.environ["SOME_KEY"] == "from_yaml"

    def test_real_env_var_wins_over_yaml_without_override(self, fake_root, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "from_real_env")
        _write_yaml(fake_root, "a.yaml", "some_key: from_yaml\n")
        load_config()
        assert os.environ["SOME_KEY"] == "from_real_env"

    def test_override_true_lets_yaml_beat_real_env(self, fake_root, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "from_real_env")
        _write_yaml(fake_root, "a.yaml", "some_key: from_yaml\n")
        load_config(override=True)
        assert os.environ["SOME_KEY"] == "from_yaml"

    def test_empty_yaml_value_does_not_export(self, fake_root):
        _write_yaml(fake_root, "a.yaml", "unset_key:\n")
        load_config()
        assert "UNSET_KEY" not in os.environ

    def test_nested_yaml_flattened_and_uppercased(self, fake_root):
        _write_yaml(fake_root, "a.yaml", "nested:\n  a:\n    b: deep_value\n")
        load_config()
        assert os.environ["NESTED_A_B"] == "deep_value"

    def test_list_value_stored_as_json_string(self, fake_root):
        import json
        _write_yaml(fake_root, "a.yaml", "list_key:\n  - one\n  - two\n")
        load_config()
        assert json.loads(os.environ["LIST_KEY"]) == ["one", "two"]

    def test_index_yaml_controls_which_files_load_and_order(self, fake_root):
        _write_yaml(fake_root, "first.yaml", "some_key: from_first\n")
        _write_yaml(fake_root, "second.yaml", "some_key: from_second\n")
        _write_yaml(fake_root, "index.yaml", "configs:\n  - second.yaml\n  - first.yaml\n")
        load_config()
        # first.yaml is processed LAST per the index order, and since both
        # define the same key with no real env var protecting it, the last
        # file processed wins (original_env check doesn't protect against
        # YAML-vs-YAML overwrites, only real pre-existing env vars)
        assert os.environ["SOME_KEY"] == "from_first"

    def test_index_yaml_referencing_missing_file_raises(self, fake_root):
        _write_yaml(fake_root, "index.yaml", "configs:\n  - does_not_exist.yaml\n")
        with pytest.raises(FileNotFoundError):
            load_config()

    def test_no_index_yaml_globs_all_yaml_files_except_index(self, fake_root):
        _write_yaml(fake_root, "a.yaml", "key_a: value_a\n")
        _write_yaml(fake_root, "b.yaml", "key_b: value_b\n")
        load_config()
        assert os.environ["KEY_A"] == "value_a"
        assert os.environ["KEY_B"] == "value_b"

    def test_missing_config_dir_does_not_raise(self, fake_root):
        # no config/ directory created at all
        load_config()  # should simply skip YAML loading, not error


class TestLoadConfigSecretsPrecedence:
    def test_env_age_fills_gap_yaml_did_not_set(self, fake_root, monkeypatch):
        state_root = fake_root / "state"
        monkeypatch.setenv("USER_STATE_ROOT", str(state_root))
        _write_yaml(fake_root, "a.yaml", "some_key: from_yaml\n")

        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / "age-key.txt").write_text("dummy-key")
        enc_path = state_root / ".env.age"
        enc_path.write_text("dummy")

        def _fake_decrypt(enc, identity):
            return {"SECRET_ONLY_KEY": "from_secrets", "SOME_KEY": "from_secrets_should_not_win"}

        monkeypatch.setattr(config_module, "_decrypt_env", _fake_decrypt)
        load_config()

        assert os.environ["SECRET_ONLY_KEY"] == "from_secrets"
        # YAML already set SOME_KEY, so the secret must NOT overwrite it
        # (without override) -- this is the "YAML wins over stale .env" guarantee
        assert os.environ["SOME_KEY"] == "from_yaml"

    def test_override_true_lets_secrets_beat_yaml(self, fake_root, monkeypatch):
        """Documents a real subtlety: with override=True, secrets are
        processed AFTER yaml and unconditionally overwrite -- so under
        override, secrets end up winning over YAML, not just real env.
        This is implementation behavior worth pinning down explicitly so
        a future refactor doesn't silently flip which one wins."""
        state_root = fake_root / "state"
        monkeypatch.setenv("USER_STATE_ROOT", str(state_root))
        _write_yaml(fake_root, "a.yaml", "some_key: from_yaml\n")

        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / "age-key.txt").write_text("dummy-key")
        (state_root / ".env.age").write_text("dummy")

        monkeypatch.setattr(
            config_module, "_decrypt_env",
            lambda enc, identity: {"SOME_KEY": "from_secrets"},
        )
        load_config(override=True)
        assert os.environ["SOME_KEY"] == "from_secrets"

    def test_relative_age_key_resolved_under_user_state_root(self, fake_root, monkeypatch):
        state_root = fake_root / "state"
        monkeypatch.setenv("USER_STATE_ROOT", str(state_root))
        monkeypatch.setenv("AGE_KEY", "my-key.txt")  # relative -- should resolve under state_root
        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / ".env.age").write_text("dummy")
        (state_root / "my-key.txt").write_text("dummy-key")

        captured = {}

        def _fake_decrypt(enc, identity):
            captured["identity"] = identity
            return {}

        monkeypatch.setattr(config_module, "_decrypt_env", _fake_decrypt)
        load_config()
        assert captured["identity"] == state_root / "my-key.txt"

    def test_absolute_age_key_override_kept_as_is(self, fake_root, monkeypatch, tmp_path_factory):
        absolute_key_dir = tmp_path_factory.mktemp("elsewhere")
        absolute_key = absolute_key_dir / "abs-key.txt"
        absolute_key.write_text("dummy-key")

        state_root = fake_root / "state"
        monkeypatch.setenv("USER_STATE_ROOT", str(state_root))
        monkeypatch.setenv("AGE_KEY", str(absolute_key))
        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / ".env.age").write_text("dummy")

        captured = {}

        def _fake_decrypt(enc, identity):
            captured["identity"] = identity
            return {}

        monkeypatch.setattr(config_module, "_decrypt_env", _fake_decrypt)
        load_config()
        assert captured["identity"] == absolute_key

    def test_falls_back_to_plaintext_dotenv_when_no_env_age(self, fake_root, monkeypatch):
        """Dev-machine fallback path -- root/.env used when no .env.age
        exists at all. Should never happen on the real Jetson deployment
        per the inline comment, but the fallback itself is still worth
        testing since it's real code that will run if someone forgets to
        set up .env.age locally."""
        pytest.importorskip("dotenv")
        state_root = fake_root / "state"
        monkeypatch.setenv("USER_STATE_ROOT", str(state_root))
        (fake_root / ".env").write_text("DOTENV_FALLBACK_KEY=from_plain_dotenv\n")
        load_config()
        assert os.environ["DOTENV_FALLBACK_KEY"] == "from_plain_dotenv"
        
    def test_no_env_age_and_no_dotenv_does_not_raise(self, fake_root):
        load_config()  # neither .env.age nor .env exists -- should just no-op silently


class TestLoadConfigCaching:
    def test_second_call_without_override_is_a_noop(self, fake_root, monkeypatch):
        _write_yaml(fake_root, "a.yaml", "some_key: first_value\n")
        load_config()
        assert os.environ["SOME_KEY"] == "first_value"

        # change the YAML and call again -- should NOT pick up the change,
        # since _LOADED guards against redundant reprocessing
        _write_yaml(fake_root, "a.yaml", "some_key: second_value\n")
        load_config()
        assert os.environ["SOME_KEY"] == "first_value"

    def test_override_true_forces_reload(self, fake_root):
        _write_yaml(fake_root, "a.yaml", "some_key: first_value\n")
        load_config()
        assert os.environ["SOME_KEY"] == "first_value"

        _write_yaml(fake_root, "a.yaml", "some_key: second_value\n")
        load_config(override=True)
        assert os.environ["SOME_KEY"] == "second_value"

    def test_loaded_flag_is_set_after_first_call(self, fake_root):
        assert config_module._LOADED is False
        load_config()
        assert config_module._LOADED is True
