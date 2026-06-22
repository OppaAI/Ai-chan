Run basic commands to verify system compatibility.

Check CUDA/cuDNN installation with `nvcc --version` and `cat /usr/include/cudnn_version.h`.

Install llama.cpp from source with CUDA support.

Load Ministral3-3B-instruct model and test initialization.

Perform a test inference with `--max_tokens 100`.

Monitor GPU memory usage with `nvidia-smi`.

Check for OOM or CUDA errors during inference.

Review logs for root causes if errors occur.