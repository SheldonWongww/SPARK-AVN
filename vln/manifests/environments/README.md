# VLN evaluation environments

`eval_environments.json` captures the critical interpreter, CUDA-wheel, model
library, simulator, and native-source versions for the six VLN baselines.
Package versions are descriptive evidence, not a replacement for the formal
run manifest required for reported results.

GOAT's pinned spaCy/Pydantic stack and English model are recorded explicitly.
GPU kernels and one-episode model execution must be rechecked when the instance
is restarted with a GPU.
