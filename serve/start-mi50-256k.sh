#!/bin/sh
# Inkling-2-Mini on two MI50s (gfx906): 256K context, expert parallel, planar Q8 dense, MTP drafter.
# The engine has FP8/F16 KV only (no Q8_0 KV), so IK2_KV=fp8.
# Measured on this tip: READY in ~42 s; plain decode ~41-46 tok/s at 262144 (q8 dense, EP, FP8 KV).
# The serve-side MTP window is off by default: it does not pay in serve (see SERVE_DRAFT). Cold start
# grows while the 16 GiB GGUF enters the page cache.
#
# Images: docs/inkling2-mini/ik2-mini-hip-256k.json's "vision" entry starts maya-vision on the CPU with
# /disk/models/mmproj-BF16.gguf.  Build it once (llama.cpp b12590) before the first start:
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#   cmake --build build-vision -j "$(nproc)" --target maya-vision
set -eu
cd "$(dirname "$0")/.."

if [ ! -f ik2/tokenizer/tokenizer.json ]; then
  ~/.venv/bin/python tools/strata_tokenizer.py \
    --gguf /disk/models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
    --chat-template tools/ik2/chat_template.jinja --check
fi

if [ ! -x build-vision/bin/maya-vision ]; then
  echo "serve: build-vision/bin/maya-vision is missing; build the image encoder as the comment above shows" >&2
  exit 1
fi

CONTEXT=${CONTEXT:-262144}
PORT=${PORT:-18000}
HOST=${HOST:-0.0.0.0}
CACHE_GB=${CACHE_GB:-14}

export HIP_VISIBLE_DEVICES=0,1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export IK2_EP=1 IK2_KV=fp8 IK2_DENSE=q8 IK2_DRAFT_CTX=65536
# The serve-side MTP window (IK2_SERVE_DRAFT), off by default.  Measured in serve on this tip: 262144
# ~31-34 against 41-45.6 tok/s plain; 131072 served ~41-42 against 45.5 plain on a 27-token prompt
# (0 of 7 drafts accepted).  The CLI at 131072 does pay (48 against 43) on high-acceptance prompts.
# SERVE_DRAFT=1 to try it.
SERVE_DRAFT=${SERVE_DRAFT:-0}
export IK2_SERVE_DRAFT=$SERVE_DRAFT
echo "serve: context $CONTEXT, cache-gb $CACHE_GB, SERVE_DRAFT=$SERVE_DRAFT"

# The config carries the defaults; a CONTEXT / CACHE_GB override gets a temporary copy.
CFG=docs/inkling2-mini/ik2-mini-hip-256k.json
if [ "$CONTEXT" != 262144 ] || [ "$CACHE_GB" != 14 ]; then
  CFG=/tmp/ik2-mini-hip-$CONTEXT-$CACHE_GB.json
  jq --arg c "$CONTEXT" --arg g "$CACHE_GB" '
    (.args | index("--max-context")) as $i | (.args | index("--cache-gb")) as $j
    | .args[$i + 1] = $c | .args[$j + 1] = $g' docs/inkling2-mini/ik2-mini-hip-256k.json > "$CFG"
fi

exec ~/.venv/bin/python serve/server.py --engine maya --config "$CFG" \
  --host "$HOST" --port "$PORT"
