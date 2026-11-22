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
| decode, 32K-token context (CLI) | | **32.2 tok/s** (31.1 ms/token) |
| prefill, 512 / 32768 tokens (CLI) | 359–389 tok/s | **414 / 442 tok/s** |
| prefill, 5152 tokens (server) | | **644.9 tok/s** |
| `ik2_kernels_test` | | **344 checks, 0 failures** (4 mma.sync tests skipped by name) |

The 32K decode figure is the final build's (`attn_split_gqa_kernel`, below). The same binary measured 33.4 tok/s
(30.0 ms/token) on a cool machine: a sustained hour of load costs ~20 % of this kernel's throughput (the
context-independent sections drift ~4 % in the same window), so treat 32.2-33.4 tok/s as the spread.

Deterministic: greedy output repeats across runs. **EP is bit-identical to a single GPU in attention, not in the
experts.** Measured on a 255-id prompt taken a token at a time (`--chunk 1`, the decode path) with both attention
dispatch modes, against one GPU: the first visit's attention output is bit-identical (`attn_out`, worst |Δ| = 0.0,
`IK2_ATTN_GQA` 0 and 1), while that visit's MoE output already differs (`out` 1.5e-4 scalar attention / 1.9e-4 GQA)
and the difference grows to 7.3 / 12.6 on the log-probs over 48 visits (102 / 107 of 145 dump files differ, expert
picks identical in 42 / 37 of the 48 visits — the picks follow the residual stream, they do not lead it). The cause
is the expert kernels, not attention: a two-card decode runs the EP block's limb (`__dp4a`) `expert_limb_mv` where
one GPU runs the row kernels (`IK2_EXPERT_LIMB` defaults to 0). It is **pre-existing**: with the branch-tip scalar
attention (`IK2_ATTN_GQA=0`) the counts are the same 102 / 145 and the greedy stream of that prompt already differs
between EP and one GPU. An earlier revision's "0 of 96 files differ, worst |Δ| = 0.0, expert picks identical" does
not reproduce on this build; the configuration it held under is not recorded here.

## Two cards: expert parallel only

`--gpus 2` **requires** `IK2_EP=1` (or `IK2_TP=1` without a drafter). The layer split is refused for this
model: its core layers (6 … 17) run three times, and a later pass reads the pass's bank of block states,
which the cross-GPU handoff does not copy — layer 12's attention then differs from a single GPU's by 0.84 at
once (logits by 18), measured. EP keeps every layer whole on GPU 0 and splits only the experts by
`ep_owner`, so both GPUs see the same arithmetic in the same order.

## The drafter: a hole in the checkpoint, not a port regression

`--draft` first proposed degenerate tokens on two cards (step 0 of a README prompt:
`62 39014 131095 71942 71942 68014 68014` = `_ axter cmds tests tests \n\n \n\n`, repeating pairs) with 8 % of
picks accepted, while the same code measured 2.63 - 4.36 tokens a window on CUDA (`c520fd1`). The port was cleared
op by op before the file was suspected:

* the drafter-only ops now have CPU references in `tests/ik2_kernels_test.cu` (`test_drafter_ops`, 16 checks, all
  green on gfx906): `mix_rms` (4 taps, 1 and 3 rows), `draft_attn` F32 and FP16 (an empty context, 512 keys, a
  wrapping 1024-row ring, 8 queries and 1), `rope_neox` (theta 1e6, 9 positions), `rnn_inputs` (the
  state | bf16(w1[prev]) | h | f_rnn layout, bit-exact), `rnn_gates` (the gate's polarity held at 0 and 1),
  `to_bf16` (bit-exact) and `silu_mul_bf16`;
* the five GEMM shapes `propose` runs - the four BF16 (`3072x2304x8`, `2304x3072x8`, `2304x9216x8`,
  `18432x2304x8`, `768x5120x1`) and the head's FP16 `200058x2304x7` - match a CPU dot to 1.9e-5 on |max| 25.5
  through hipBLAS `GemmEx` (H2 closed);
* the target's verify window is bit-clean: `--chunk 2` and `--chunk 8` (every prompt position through a T = 2 / 8
  forward) reproduce the plain stream's 16 tokens exactly, and the `--draft` stream follows plain decode's (one
  near tie flips it at 255, below);
* `f_rnn` (the Markov cell's context row, `set_f_rnn`'s `mix_rms`) came out **all zeros** although the taps and the
  `fuse_weight` row were normal - which pointed at the weights, not the kernels.

The checkpoint was a partial write. `/disk/models/mtp.safetensors` had one 136 MB unwritten run in its data section
(bytes 423,861,938 .. 559,806,276: `layers.2`'s `q/k/v/o_proj`, `k/v_proj_ctx`, `up_proj` and its norms,
`markov_head.dflare_norm` (so `f_rnn` was zero), `markov_head.hidden_out.weight/bias` (so the Markov fold added
nothing) and the first 14.8 MB of `markov_head.markov_w1`). Everything outside the hole was intact and matched
`/disk/models/dspark/dspark-Inkling-2-Mini-BF16.gguf` - the BF16 conversion of the same checkpoint - for all 54
tensors, so the hole was rebuilt from it byte for byte (the repaired file now agrees with that GGUF at every byte;
sha256 `7c3b6513...a1d2`). The corrupt original is kept as `/disk/models/mtp.safetensors.hole-20261122`.

After the repair, at 255 / 512 / 32768 tokens (CLI, `--gen`, `IK2_DRAFT_WIDTH=5` unless noted):

| prompt | windows | tokens a window | drafts accepted | dr[0] | drafting + verifying | tok/s (plain) |
|---|---|---|---|---|---|---|
| 255, width 8 | 27 | **2.33** | 43 of 189 (23 %) | 59 % | 16.9 + 197.1 ms | 10.9 (43.9) |
| 255 | 28 | 2.25 | 27 of 112 (24 %) | 61 % | 15.4 + 147.4 ms | 13.8 (43.9) |
| 512 (README tail), width 8 | 18 | **3.50** | 45 of 126 (36 %) | 72 % | 17.2 + 195.7 ms | 16.4 (43.8) |
| 512 | 14 | 2.21 | 13 of 56 (23 %) | 57 % | 15.7 + 152.2 ms | 13.2 (43.8) |
| 32768 | 13 | 2.38 | 14 of 52 (27 %) | 62 % | 21.0 + 140.5 ms | 14.8 (34.5) |

The proposals continue the text (a window that accepts all four: `147403 62 146312 146024`), and the acceptance
tracks the text: the README's shell-command tail is high entropy. The greedy stream is the target's: identical to
plain decode at 32768 and to position 44 at 255, where 159542 / 152804 tie at 27.0161 / 27.0121 - a window's
different summation order; `IK2_DRAFT_WIDTH=2` reproduces the plain stream exactly. `IK2_DRAFT_CTX=2048` changes
nothing at these depths (both caches hold the whole context).

What is left is the window's cost, not its correctness: 147 - 197 ms a window against 22.6 ms a plain token, i.e.
about a plain token a row, because `forward()` runs all 48 visits for the window's rows. A scalar block attention
(the CUDA `attn_block_mma`'s job) is what a 2 - 3x faster window needs.

## Disabled on gfx906

* **Tensor-core attention and experts** (`mma.sync m16n8k16`, Hopper `wgmma`, `cp.async`, `ldmatrix`,
  `q8_tile`, the grouped-block kernels): PTX, compiled out under `STRATA_USE_HIP`; their launchers abort with
  `maya-ik2: <fn>: tensor-core path not built for HIP/gfx906`. The GQA split attention
  (`attn_split_gqa_kernel`), the scalar split attention it falls back to, and the limb (`__dp4a`) expert kernels
  run instead. `IK2_FLASH=1`, `IK2_ATTN_MMA=1`, `IK2_BLOCK=1`, `IK2_EXPERT_MMA=1`
  are refused at startup rather than run.
* **The block path** (a verify window's attention, `attn_block_mma`) and CUDA graphs: off (`IK2_GRAPH=0`).
* **The drafter through the general prompt path** (`--draft`): correct on two cards with EP after the checkpoint
  repair above (its proposals continue the text, 2.2 - 3.5 tokens a window), but a verify window runs the whole
  48-visit loop - the block path is mma.sync, compiled out here - so a window of B costs about B plain tokens and
  the result is slower than plain decoding (13.8 tok/s against 43.9 at 255 tokens). Off by default; the
  sections above and below it describe the state and what is left.
* **Vision**: the mmproj file exists but the vision encoder is not built for HIP here.

## Cold start

The 14.7 GiB of experts are read once into pinned host memory, then each GPU's cache is filled with the
experts it owns. On this machine the read runs at ~0.4 GiB/s when the page cache is cold (37 s) and ~2 s when
warm; the per-GPU fill is 0.6 s; `READY` follows. Keep the GGUF page-cached for serving sessions.

## Open work

1. The token loop's per-kernel pipeline gap: the attention phase measures ~13.8 ms a token against ~6 ms of
   kernel service in isolation, and is not the context footprint (8192 vs 131072 changes nothing). The
   device-side pick split removed the per-layer host round trip and its packs (**+0.6–0.8 ms a token**:
   43.4 → 44.2 tok/s at 255 tokens, 31.0 → 31.7 at 32K — the table above has the current figures). What remains
   is the gap between kernels: graph capture would attack it, but capturing across two devices is the blocker;
   per-layer, two-device graphs are the untested alternative.
2. `attn_res_rms_f16` costs ~25 µs a call with an idle queue (one block of 512 threads, latency-bound; the
   isolated pipelined figure is 11.7 µs). The ~10 µs around each of the two calls a visit — and the same gap
   on every other small kernel — is the pipeline overhead above, not kernel time.
3. Deep-context attention: 10.05 -> 6.3-7.5 ms a token at 32K with the GQA split kernel (`attn_split_gqa_kernel`
   in `src/ik2/ik2_kernels.cu`): a block a K / V head, its `H / Hkv` (3 here) query heads in one pass over the
   keys, so each K / V row is read and FP8-decoded once instead of three times, 16-byte loads, the E4M3 decode
   from a 256-entry LDS table, and the sliding layers' ring index incremental instead of a per-key divide.
   `IK2_ATTN_GQA=0` runs the previous scalar `attn_split_kernel` (bit-identical to the branch tip). `test_attn_gqa`
   in `tests/ik2_kernels_test.cu` compares the two on the same K / V for H 24 / KV 8 (full, ring), H 48 / KV 8
   (GQA 6, masked ring) and an FP16 cache, and checks that a shape no GQA kernel takes (H 16 / KV 4) stays on the
   scalar path bit for bit; `attn_gqa_mode()` is the hook it switches them with. That is the 3x fewer K / V reads;
   what is left is issue count and latency (IPC 0.57 against the scalar path's 0.94): 6 `expf` a key (13
   instructions each, kept for bit-exactness), three 32-lane xor butterflies (15 `ds_bpermute` + waits), and the
   LDS table read feeding the dot. Next shapes: a warp taking 4 keys with 8-lane reduces (9 shuffles per 4 keys),
   or `exp2f` (2 instructions) if a last-bit change is acceptable.
4. Planar Q8 dense (`IK2_DENSE=q8`) works with EP and halves the dense bytes a token reads (42.9 → 43.2
   tok/s) but a prefill chunk then dequantizes every projection per pass (421 → 209 tok/s): opt-in only.
5. The drafter's verify window: the scalar block attention above (a window costs a plain token a row
   through `forward()`), and a sampler-aware accept rule before it can reach the server.
