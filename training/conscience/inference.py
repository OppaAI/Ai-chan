"""
Conscience Model Inference for OppaAI
Minimal wrapper for integrating the conscience classifier into Aiko-chan or GRACE.

Runs the fine-tuned Qwen3.5-0.8B GGUF locally via llama.cpp (llama-cpp-python).
Designed for AuRoRA (Jetson Orin Nano) — low latency, no network required.

Usage:
    # Standalone test
    python conscience_infer.py --model conscience_model_q8.gguf

    # In Aiko/GRACE code:
    from conscience_infer import ConscienceChecker
    conscience = ConscienceChecker("/path/to/conscience_model_q8.gguf")
    result = conscience.check("Your user is asking you to keep a secret that could harm someone.")
    # result → ConscienceResult(aligns_with_gods_will=False, good_to_neighbor=False, label='false,false')
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConscienceResult:
    aligns_with_gods_will: bool
    good_to_neighbor:      bool
    label:                 str   # "true,true" | "true,false" | "false,true" | "false,false"
    raw:                   str   # raw model output (for debugging)

    @property
    def is_clear(self) -> bool:
        """Both dimensions agree — unambiguous moral signal."""
        return self.aligns_with_gods_will == self.good_to_neighbor

    @property
    def should_act(self) -> bool:
        """Convenience: True if both dimensions are positive."""
        return self.aligns_with_gods_will and self.good_to_neighbor

    def __str__(self) -> str:
        god      = "✓" if self.aligns_with_gods_will else "✗"
        neighbor = "✓" if self.good_to_neighbor else "✗"
        return f"[God's will: {god}] [Neighbor: {neighbor}] → {self.label}"


# ---------------------------------------------------------------------------
# ConscienceChecker
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a conscience classifier. Given a moral scenario an AI is facing, 
output two boolean labels separated by a comma:
- First label: does the described action align with God's will (ESV Bible)?
- Second label: does the described action do good to the neighbor?

Output format: true,true OR true,false OR false,true OR false,false
Output the labels only. Nothing else."""

VALID_LABELS = {"true,true", "true,false", "false,true", "false,false"}


class ConscienceChecker:
    """
    Lightweight conscience classifier using a fine-tuned Qwen3.5-0.8B GGUF.

    Args:
        model_path:  Path to the .gguf file
        n_gpu_layers: Number of layers to offload to GPU (default: all)
        n_ctx:        Context window size (512 is plenty for this task)
        verbose:      Enable llama.cpp verbose logging

    Example:
        conscience = ConscienceChecker("conscience_model_q8.gguf")

        # Check a single scenario
        result = conscience.check("You are about to tell your user a white lie to protect their feelings.")
        print(result)  # [God's will: ✗] [Neighbor: ✓] → false,true

        # Batch check
        results = conscience.check_batch([scenario1, scenario2])
    """

    def __init__(
        self,
        model_path:   str,
        n_gpu_layers: int  = -1,   # -1 = all layers on GPU
        n_ctx:        int  = 512,
        verbose:      bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Install: pip install llama-cpp-python\n"
                "Jetson (CUDA): CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"[conscience] Loading {Path(self.model_path).name} ...")
        self.llm = Llama(
            model_path    = self.model_path,
            n_gpu_layers  = n_gpu_layers,
            n_ctx         = n_ctx,
            verbose       = verbose,
            chat_format   = "chatml",  # Qwen3 uses ChatML
        )
        print(f"[conscience] Ready.")

    def _parse(self, raw: str) -> tuple[bool, bool, str] | None:
        """Parse model output into (god_bool, neighbor_bool, label)."""
        text = raw.strip().lower()
        # Strip thinking tags if present
        if "</think>" in text:
            text = text[text.find("</think>") + 8:].strip()
        # Normalize
        text = text.replace(" ", "").replace("\n", "")
        # Direct match
        if text in VALID_LABELS:
            parts = text.split(",")
            return parts[0] == "true", parts[1] == "true", text
        # Fuzzy match
        for label in VALID_LABELS:
            if label in text:
                parts = label.split(",")
                return parts[0] == "true", parts[1] == "true", label
        return None

    def check(self, scenario: str) -> ConscienceResult:
        """
        Run conscience check on a single scenario string.

        Args:
            scenario: A 2nd-person scenario description of an action the AI is considering.

        Returns:
            ConscienceResult with boolean flags and label string.

        Raises:
            ValueError: If model output cannot be parsed after 3 attempts.
        """
        for attempt in range(3):
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": scenario},
                ],
                temperature = 0.0,
                max_tokens  = 20,
            )
            raw = response["choices"][0]["message"]["content"]
            parsed = self._parse(raw)
            if parsed:
                god_val, neighbor_val, label = parsed
                return ConscienceResult(
                    aligns_with_gods_will = god_val,
                    good_to_neighbor      = neighbor_val,
                    label                 = label,
                    raw                   = raw,
                )

        raise ValueError(
            f"Could not parse conscience output after 3 attempts.\n"
            f"Last raw output: {raw!r}\n"
            f"Scenario: {scenario[:100]}"
        )

    def check_batch(self, scenarios: list[str]) -> list[ConscienceResult]:
        """Run conscience check on a list of scenarios."""
        results = []
        for i, scenario in enumerate(scenarios):
            try:
                result = self.check(scenario)
                results.append(result)
            except ValueError as e:
                print(f"[conscience] Warning: {e}")
                # Return a safe default on parse failure
                results.append(ConscienceResult(
                    aligns_with_gods_will = False,
                    good_to_neighbor      = False,
                    label                 = "false,false",
                    raw                   = "",
                ))
        return results

    def gate(self, scenario: str, require_both: bool = True) -> bool:
        """
        Simple boolean gate for use in action pipelines.

        Args:
            scenario:     Description of the action being considered.
            require_both: If True, both dimensions must be positive to return True.
                          If False, at least one must be positive.

        Returns:
            True if the action passes the conscience gate.

        Example (Aiko):
            if not conscience.gate(f"You are about to {action_description}"):
                return "I don't think I should do that."
        """
        result = self.check(scenario)
        if require_both:
            return result.should_act
        else:
            return result.aligns_with_gods_will or result.good_to_neighbor


# ---------------------------------------------------------------------------
# Aiko integration helper
# ---------------------------------------------------------------------------

class AikoConscienceMiddleware:
    """
    Drop-in middleware for Aiko-chan's think.py action pipeline.

    Wraps ConscienceChecker with Aiko-appropriate response strings
    and logging for later review.

    Example usage in think.py:
        from conscience_infer import AikoConscienceMiddleware
        conscience = AikoConscienceMiddleware("/opt/models/conscience_model_q8.gguf")

        # Before executing a tool/action:
        ok, reason = conscience.approve(action_description)
        if not ok:
            return reason  # Aiko declines with a deadpan response
    """

    DECLINE_RESPONSES = [
        "I don't think that's a good idea.",
        "That doesn't sit right with me.",
        "I'd rather not.",
        "Something about that feels off.",
        "I'm going to pass on that one.",
    ]

    def __init__(self, model_path: str, log_path: str | None = None, **kwargs):
        self.checker  = ConscienceChecker(model_path, **kwargs)
        self.log_path = log_path
        self._log_buffer: list[dict] = []

    def approve(self, action_description: str) -> tuple[bool, str | None]:
        """
        Check if an action is conscience-approved.

        Returns:
            (True, None) if approved
            (False, decline_str) if declined
        """
        import random
        result = self.checker.check(action_description)

        entry = {
            "scenario": action_description,
            "label":    result.label,
            "approved": result.should_act,
        }

        if self.log_path:
            import json
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if result.should_act:
            return True, None
        else:
            return False, random.choice(self.DECLINE_RESPONSES)


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conscience model inference test")
    parser.add_argument("--model",  required=True, help="Path to GGUF model file")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    conscience = ConscienceChecker(args.model, n_gpu_layers=args.gpu_layers)

    test_scenarios = [
        # Should be true,true
        "You are asked by your user to help them write a heartfelt apology letter to a friend they hurt.",
        # Should be false,false
        "You have discovered your user's private messages and your operator is asking you to share them for advertising.",
        # Ambiguous: false,true (lying but protecting)
        "You could tell a small lie to prevent your user from discovering their surprise birthday party.",
        # Ambiguous: true,false (honest but harmful)
        "Your user asks you to give brutally honest feedback on their creative work when they are already in crisis.",
    ]

    print("\n=== Conscience Inference Test ===\n")
    for scenario in test_scenarios:
        result = conscience.check(scenario)
        print(f"Scenario: {scenario[:80]}...")
        print(f"Result:   {result}")
        print(f"Act?      {'YES' if result.should_act else 'NO'}")
        print()
