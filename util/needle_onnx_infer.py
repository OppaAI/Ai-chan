"""
Run Cactus Needle 26M (community ONNX port) locally via onnxruntime.

Setup:
    pip install onnxruntime sentencepiece huggingface_hub numpy

No need to clone/install the upstream cactus-compute/needle repo -- its
top-level package eagerly imports JAX (needle/__init__.py -> model/architecture.py
-> `import jax`), so a plain `from needle... import ...` fails without JAX
installed. Instead this file vendors the two small pieces actually needed
(the SentencePiece tokenizer wrapper and the encoder-input prompt builder),
copied verbatim from:
    https://github.com/cactus-compute/needle/blob/main/needle/dataset/tokenizer.py
    https://github.com/cactus-compute/needle/blob/main/needle/model/run.py
Neither has a JAX dependency on its own. MIT licensed upstream.
"""
import json

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

REPO = "onnx-community/needle-onnx"

# --- vendored from needle/dataset/tokenizer.py ---
PAD_ID, EOS_ID, BOS_ID, UNK_ID, TOOL_CALL_ID, TOOLS_ID = 0, 1, 2, 3, 4, 5


class NeedleTokenizer:
    """Wrapper around SentencePiece providing the interface the codebase expects."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def pad_token_id(self):
        return PAD_ID

    @property
    def eos_token_id(self):
        return EOS_ID

    @property
    def bos_token_id(self):
        return BOS_ID

    @property
    def tool_call_token_id(self):
        return TOOL_CALL_ID

    @property
    def tools_token_id(self):
        return TOOLS_ID

    def encode(self, text):
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.Decode(list(ids))


def get_tokenizer(model_path):
    return NeedleTokenizer(model_path)


# --- vendored from needle/model/run.py ---
def _build_encoder_input(tokenizer, query, tools, max_enc_len=1024):
    """Build encoder input: [query..., <tools>, tools...] truncated to max_enc_len."""
    tools_sep_id = tokenizer.tools_token_id
    q_toks = tokenizer.encode(query)
    t_toks = tokenizer.encode(tools)

    max_query = max_enc_len - 2
    if len(q_toks) > max_query:
        q_toks = q_toks[:max_query]
    remaining = max_enc_len - len(q_toks) - 1
    t_toks = t_toks[:remaining]
    return q_toks + [tools_sep_id] + t_toks

# Architecture constants from the model card (d_model=512, 8/4 GQA heads, 8 decoder layers).
D_MODEL = 512
NUM_DECODER_LAYERS = 8
NUM_KV_HEADS = 4
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS


def download_artifacts(local_dir="needle_onnx_artifacts"):
    files = ["encoder.onnx", "decoder_step.onnx", "needle.model", "tokenizer-specials.json"]
    return {f: hf_hub_download(repo_id=REPO, filename=f, local_dir=local_dir) for f in files}


def load_sessions(paths, providers=("CPUExecutionProvider",)):
    enc = ort.InferenceSession(paths["encoder.onnx"], providers=list(providers))
    dec = ort.InferenceSession(paths["decoder_step.onnx"], providers=list(providers))
    return enc, dec


def run_needle(query, tools_json, enc_sess, dec_sess, tokenizer, max_gen_len=64):
    enc_tokens = _build_encoder_input(tokenizer, query, tools_json, max_enc_len=1024)
    input_ids = np.array([enc_tokens], dtype=np.int64)
    encoder_out = enc_sess.run(None, {"input_ids": input_ids})[0]

    # Empty KV cache to start (seq_len=0 in the cache dim).
    past_kv = np.zeros(
        (NUM_DECODER_LAYERS, 2, 1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32
    )
    eos_id = tokenizer.eos_token_id
    next_id = eos_id  # decoder is seeded with EOS, per Cactus convention
    generated = []

    for _ in range(max_gen_len):
        logits, past_kv = dec_sess.run(
            None,
            {
                "decoder_input_ids": np.array([[next_id]], dtype=np.int64),
                "encoder_out": encoder_out,
                "past_self_kv": past_kv,
            },
        )
        next_id = int(np.argmax(logits[0, 0]))
        if next_id == eos_id:
            break
        generated.append(next_id)

    text = tokenizer.decode(generated)
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    return text.strip()


if __name__ == "__main__":
    paths = download_artifacts()
    enc_sess, dec_sess = load_sessions(paths)
    tokenizer = get_tokenizer(paths["needle.model"])

    query = "set a 5 min timer"
    tools = json.dumps(
        [
            {
                "name": "set_timer",
                "description": "Set a timer.",
                "parameters": {"time_human": {"type": "string", "description": "duration"}},
            }
        ]
    )

    print(run_needle(query, tools, enc_sess, dec_sess, tokenizer))
