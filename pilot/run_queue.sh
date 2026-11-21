#!/usr/bin/env bash
# Runs the pilot measurements in priority order (CPU-only, resumable:
# chapters already present in results/llm_bits.jsonl are skipped).
# Secondary experiments use only the six New Testament chapters of the sample
# so that the whole queue fits into about an hour on four CPU cores.
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/data/hf}" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# Leave one core free: oversubscribed OpenMP threads slow the model down 10-30x.
export OMP_WAIT_POLICY=PASSIVE
THREADS="${THREADS:-3}"
PY="${PY:-.venv/bin/python}"
S=Qwen/Qwen3.5-0.8B-Base
M=Qwen/Qwen3.5-2B-Base
run() { "$PY" pilot/llm_bits.py --threads "$THREADS" "$@" 2>&1 | grep --line-buffered -v -e Warning -e "Loading weights" -e "falling back" -e "Fetching"; }

# 1. breadth: every translation with the small base model, no cross-lingual context
run --model $S --mode uncond --translations eng-kjv2006,deu1912,engwebp,fraLSG,russyn,cmn-cu89s,swhulb,aai,grcbyz,hboWLC,hin2017,latVUC,deuelbbk
# 2. same verse of an already-decoded translation in the context
run --model $S --mode cond --nt-only --pivot eng-kjv2006 --translations deu1912,aai,engwebp,cmn-cu89s,swhulb
run --model $S --mode cond --nt-only --pivot deu1912 --translations deuelbbk
# 3. model-size trend
run --model $M --mode uncond --nt-only --translations eng-kjv2006,deu1912,aai
# 4. memorisation probe on famous chapters
run --model $S --mode uncond --sample famous --translations eng-kjv2006,deu1912
run --model $M --mode uncond --sample famous --translations eng-kjv2006
# 5. chat-tuned vs. base checkpoint
run --model Qwen/Qwen3.5-0.8B --mode uncond --nt-only --translations eng-kjv2006,deu1912
# 6. larger model with pivot on the low-resource language
run --model $M --mode cond --nt-only --pivot eng-kjv2006 --translations aai
echo QUEUE_DONE
