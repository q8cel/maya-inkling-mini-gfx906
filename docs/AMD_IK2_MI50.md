# maya-ik2 on 2 × MI50 (gfx906, HIP)

This describes branch `hip/gfx906-mi50`: the port itself is commit `0d14c59942` (stage 1 plus the two-card
expert-parallel work) and the branch tip adds the device-side pick split, this document and the README line.

The Inkling-2 engine (`src/ik2`) built for AMD Instinct MI50 / MI60 (Vega 20, `gfx906`, wave64) with ROCm 6.4.
Two cards run the experts in parallel (`IK2_EP=1`); every layer's attention, dense weights, K / V cache and taps
are GPU 0's. The HIP compatibility layer itself (wave64 mapping, `__dp4a`, `__byte_perm`, `__shfl_*_sync`, the
4 KB-parameter and 64 KiB-LDS differences) is documented in [docs/MI50_VALIDATION.md](MI50_VALIDATION.md).

## Build

```sh
cmake -S . -B build-hip-gfx906 \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp      # or omit: fetch the pinned commit in third_party/ggml/VERSION.txt
cmake --build build-hip-gfx906 --target ik2_kernels maya-ik2 ik2_kernels_test
HIP_VISIBLE_DEVICES=0 HSA_FORCE_FINE_GRAIN_PCIE=1 ./build-hip-gfx906/ik2_kernels_test
```

`STRATA_PREFILL_MMQ` is required by the project's HIP configuration and pulls in `STRATA_NATIVE_EXPERTS`
(ggml), so an ik2-only build still configures ggml. The ik2 sources compile with `-ffp-contract=off` (the
kernels' bit-exact checks); one device helper (`attn_flash_range`) opts back into FMA, which a long FP8
cache's tolerance needs.

## Run

```sh
# a token-id prompt, greedy, 64 tokens
HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1 IK2_EP=1 IK2_KV=fp8 \
  ./build-hip-gfx906/maya-ik2 --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --gpus 2 \
  --tokens @prompt.txt --gen 64 --max-context 131072 --cache-gb 14
```

## Server

```sh
~/.venv/bin/python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
    --chat-template tools/ik2/chat_template.jinja --check
~/.venv/bin/python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-hip.json
```

`docs/inkling2-mini/ik2-mini-hip.json` is the two-card HIP config: `--gpus 2`, `--cache-gb 14`,
`--max-context 131072`, env `IK2_EP=1`, `IK2_KV=fp8`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, tokenizer
`ik2/tokenizer` (model-derived: keep it untracked). At a 131072 context the fit is GPU 0 7.26 GiB of its
experts + 3.4 GiB dense + 2.2 GiB KV + work ≈ 14.4 of 15.8 GiB free, GPU 1 7.45 GiB of experts.

## Measured (2 × MI50, 16 GiB each)

| | llama.cpp (baseline) | maya-ik2, HIP |
|---|---|---|
| decode, 255-token context (CLI, 64 tokens) | 30.4 tok/s | **44.2 tok/s** (22.6 ms/token) |
| decode, 255-token context (server, 51-token prompt) | | **45.5 tok/s** |
| decode, 5.1K-token context (server) | | 39.5 tok/s |
| decode, 32K-token context (CLI) | | **31.7 tok/s** (31.6 ms/token) |
| prefill, 512 / 32768 tokens (CLI) | 359–389 tok/s | **414 / 442 tok/s** |
| prefill, 5152 tokens (server) | | **644.9 tok/s** |
| `ik2_kernels_test` | | **322 checks, 0 failures** (4 mma.sync tests skipped by name) |

Deterministic: greedy output is the same across runs and modes. EP is bit-identical to a single GPU: 0 of 96
per-layer dump files differ (48 visits × `out` + `attn_out`), worst |Δ| = 0.0, expert picks identical.

## Two cards: expert parallel only

`--gpus 2` **requires** `IK2_EP=1` (or `IK2_TP=1` without a drafter). The layer split is refused for this
model: its core layers (6 … 17) run three times, and a later pass reads the pass's bank of block states,
which the cross-GPU handoff does not copy — layer 12's attention then differs from a single GPU's by 0.84 at
once (logits by 18), measured. EP keeps every layer whole on GPU 0 and splits only the experts by
`ep_owner`, so both GPUs see the same arithmetic in the same order.

## Disabled on gfx906

* **Tensor-core attention and experts** (`mma.sync m16n8k16`, Hopper `wgmma`, `cp.async`, `ldmatrix`,
  `q8_tile`, the grouped-block kernels): PTX, compiled out under `STRATA_USE_HIP`; their launchers abort with
  `maya-ik2: <fn>: tensor-core path not built for HIP/gfx906`. The scalar split attention and the limb
  (`__dp4a`) expert kernels run instead. `IK2_FLASH=1`, `IK2_ATTN_MMA=1`, `IK2_BLOCK=1`, `IK2_EXPERT_MMA=1`
  are refused at startup rather than run.
* **The block path** (a verify window's attention, `attn_block_mma`) and CUDA graphs: off (`IK2_GRAPH=0`).
* **The drafter** (`--draft`): loads and runs on two cards with EP, but its proposals are degenerate (8 % of
  picks accepted) and a verify window costs about a plain token per row, so it is slower than plain decoding
  (7.8 tok/s vs 43). Needs a correctness pass on `ik2::draft::propose`.
* **Vision**: the mmproj file exists but the vision encoder is not built for HIP here.

## Cold start

The 14.7 GiB of experts are read once into pinned host memory, then each GPU's cache is filled with the
experts it owns. On this machine the read runs at ~0.4 GiB/s when the page cache is cold (37 s) and ~2 s when
warm; the per-GPU fill is 0.6 s; `READY` follows. Keep the GGUF page-cached for serving sessions.

## Open work

1. The token loop's per-kernel pipeline gap: the attention phase measures ~13.8 ms a token against ~6 ms of
   kernel service in isolation, and is not the context footprint (8192 vs 131072 changes nothing). The
   device-side pick split removed the per-layer host round trip and its packs (**+0.6–0.8 ms a token**:
   43.4 → 44.2 tok/s at 255 tokens, 31.0 → 31.7 at 32K). What remains is the gap between kernels: graph
   capture would attack it, but capturing across two devices is the blocker; per-layer, two-device graphs are
   the untested alternative.
2. `attn_res_rms_f16` costs ~25 µs a call with an idle queue (one block of 512 threads, latency-bound; the
   isolated pipelined figure is 11.7 µs). The ~10 µs around each of the two calls a visit — and the same gap
   on every other small kernel — is the pipeline overhead above, not kernel time.
3. The deep-context attention kernel: 8.1 ms a token at 32K (63 GB/s effective; 24 query heads re-read each
   K / V head 3×). A wave64 variant with 16-byte K/V loads is the likely shape.
4. Planar Q8 dense (`IK2_DENSE=q8`) works with EP and halves the dense bytes a token reads (42.9 → 43.2
   tok/s) but a prefill chunk then dequantizes every projection per pass (421 → 209 tok/s): opt-in only.
5. The drafter's `propose`, and a sampler-aware accept rule before it can reach the server.
