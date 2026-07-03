import sys
import threading
import types
import unittest

# Keep these focused unit tests independent of optional runtime dependencies that
# are loaded by the application-wide config/log bootstrap during import.
yaml_stub = types.SimpleNamespace(safe_load=lambda _text: {})
dotenv_stub = types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None)
sys.modules.setdefault("yaml", yaml_stub)
sys.modules.setdefault("dotenv", dotenv_stub)

from core.speak import AikoSpeak


class _RemoteOwner:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0
        self.payloads = []

    def broadcast_audio_bytes(self, payload: bytes) -> None:
        self.payloads.append(payload)

    def has_remote_listener(self) -> bool:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


class SpeakRemoteAudioTests(unittest.TestCase):
    def _speaker(self) -> AikoSpeak:
        speaker = AikoSpeak.__new__(AikoSpeak)
        speaker.local_playback = True
        speaker._audio_sink = None
        speaker._stop_flag = threading.Event()
        speaker._first_audio_fired = threading.Event()
        speaker._first_audio_callback = None
        return speaker

    def test_has_remote_listener_reflects_bound_webui_sink(self):
        speaker = self._speaker()
        owner = _RemoteOwner(result=True)
        speaker.set_audio_sink(owner.broadcast_audio_bytes)

        self.assertTrue(speaker._has_remote_listener())
        self.assertEqual(owner.calls, 1)

    def test_has_remote_listener_is_false_without_checker_or_on_error(self):
        speaker = self._speaker()
        speaker.set_audio_sink(lambda payload: None)
        self.assertFalse(speaker._has_remote_listener())

        owner = _RemoteOwner(exc=RuntimeError("boom"))
        speaker.set_audio_sink(owner.broadcast_audio_bytes)
        self.assertFalse(speaker._has_remote_listener())

    def test_play_wav_bytes_skips_local_playback_when_browser_connected(self):
        speaker = self._speaker()
        owner = _RemoteOwner(result=True)
        speaker.set_audio_sink(owner.broadcast_audio_bytes)
        speaker._wav_duration = lambda payload: 0.0

        def fail_if_local_playback_is_attempted():
            raise AssertionError("local playback should be skipped")

        speaker._load_sd = fail_if_local_playback_is_attempted

        speaker._play_wav_bytes(b"not-a-real-wav")

        self.assertEqual(owner.payloads, [b"not-a-real-wav"])
        self.assertEqual(owner.calls, 1)


if __name__ == "__main__":
    unittest.main()
