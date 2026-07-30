""" Backward-compatible re-exports — prefer sensory.listen.correct_asr_text. """
from sensory.listen import correct_asr_text, correction_pairs  # noqa: F401


def install_asr_correct_hooks() -> None:
    """No-op: correction runs inside AikoListen._transcribe."""
    return None
