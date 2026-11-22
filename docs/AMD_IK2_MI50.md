# maya-ik2 on 2 × MI50 (gfx906, HIP)

This describes branch `hip/gfx906-mi50`: the port itself is commit `0d14c59942` (stage 1 plus the two-card
expert-parallel work) and the branch tip adds the device-side pick split, this document and the README line.
The branch is synced with upstream `e819852` (the `c520fd1` base plus eight commits: the Windows/MSVC and 8 GB
card paths, the expert-fit note, the tokenizer test names and a single-token EP expert MMA path); the merge kept
the HIP guards described below.

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

Final state of the branch (`IK2_EP=1`, FP8 KV, `IK2_DRAFT_ADAPT=0`, a shared box with load ~2 - 7). The dense mode
a row was taken with matters and is named. On **two** devices the fit check leaves the dense weights **FP16**: its
planar-Q8 auto-selection sits in the single-GPU branch, so `IK2_DENSE` unset means FP16 here, and `IK2_DENSE=q8` is
what the 44.9 row (and the 32K row) used - it overrides the mode either way and costs the *window* ~20 ms a step
(39.2 against 48.1 tok/s), so a loader with `--draft` should leave it unset (`IK2_DENSE=f16` forces FP16 explicitly).
The spec rows are FP16:

| | llama.cpp (baseline) | maya-ik2, HIP |
|---|---|---|
| decode, 255-token context (CLI, 64 tokens) | 30.4 tok/s | **43.1 tok/s** (42.73 - 43.51, 5 runs, FP16 default; 44.9 with `IK2_DENSE=q8`) |
| decode, 32K-token context (CLI) | | **35.1 tok/s** (35.05 / 35.08, `IK2_DENSE=q8`) |
| prefill, 512 / 32768 tokens (CLI) | 359–389 tok/s | **429 / 440 tok/s** |
| `--draft`, 512-id README, width 5, `IK2_DRAFT_ADAPT=0` | | **48.1 - 49.6 tok/s** (48.9 mean of 5), 2.95 - 3.00 tokens a window, 15.6 + 44.9 - 46.0 ms (1.13x the 43.1 plain) |
| `ik2_kernels_test` | | **357 checks, 0 failures** |

The 32K decode figure is the final build's (`attn_split_gqa_kernel`, below) and is paired: 35.05 and 35.08 tok/s
(28.5 ms/token), two runs back to back on the shared box (load ~2, 24 - 29 C, sclk 1801 MHz, mclk 1000). On an idle
box with the GGUF page-cached the same binary has measured 34.4 - 34.6; a sustained hour of load costs ~20 % of this
kernel's throughput (the context-independent sections drift ~4 % in the same window), so the 32K range to trust is
**34.4 - 35.1 tok/s**, 35.05/35.08 being the final paired figure. The older 32.2 - 33.4 spread was the
pre-GQA-split-and-merge build and is not this one.

### After the upstream sync (`e819852`)

The merge re-applied the HIP guards on upstream's new code; all three targets build for gfx906 and
`ik2_kernels_test` is **344 checks, 0 failures** at that commit (the tip is **357** after the two-level attention
tests, five of which `test_attn_two_level` adds). The 345-check log quoted during the sync was not the committed
tip: it included an uncommitted `attn window queries / per-query: H24 KV8 8 queries 5000 keys` check (a window
experiment; `/tmp/kt_new.log`). A clean build of `c11d62e` runs 344 `ok` checks with exactly this tree's list, so
no check was lost in the merge. Newly disabled by HIP: upstream's single-token EP expert fast path
now calls `experts_gate_up_swiglu_mma` / `experts_down_mma` (on unless `IK2_EXPERT_MMA=0`) when `experts_mma_fits`
accepts the expert types; on HIP that predicate returns false, so the EP token path keeps the limb (`__dp4a`)
kernels. Both MMA entry points are stubs that abort with a clear message, so naming them directly fails loudly
instead of launching nothing. Upstream's other additions are device-agnostic and run unchanged on HIP:
`IK2_CPU_Q8`, `IK2_LAZY_CPU` / `_MAX` / `_PER` / `_STREAM` (single GPU), `IK2_ROOM_MIB`, the drafter-aware fit
message and the `resident=` field of the server's `INFO` line. Upstream's new `n_gpu > 1 && IK2_DENSE` refusal is
CUDA-only here, so the port's documented EP + planar-Q8 opt-in still runs: `IK2_DENSE=q8` on the 512-id prompt
measured 43.78 tok/s decode and 366.9 tok/s prefill.

Measured after the merge (two runs each, same box and commands as the table above):

| | run A | run B |
|---|---:|---:|
| decode, 512-id README prompt, 64 tokens | 43.02 tok/s | 43.18 tok/s |
| prefill of that prompt | 363.6 tok/s | 362.4 tok/s |
| decode, 32768-id prompt, 32 tokens | 34.54 tok/s | 34.59 tok/s |
| prefill of that prompt | 435.8 tok/s | 441.7 tok/s |

That is inside the pre-sync spread (the table's 42.7 tok/s on 511 README ids; the 32K logs above range 30.2-34.8
tok/s), so the merge did not regress plain decode; the 32768-id run repeated the same 31 ids both times
(`69201 69057 220 ...`). The 512-id pair above differed under the pre-fix nondeterminism described below.

Prefill A/B, pre-sync `c11d62e` against the merge, three alternating runs each on the same 512-id prompt and flags:
pre-sync 361.6 / 363.7 / 362.9 tok/s, merged 358.7 / 354.1 / 364.8 tok/s - no regression. The older 414 tok/s
512-token row was a different prompt file (`ik2-prompt.txt`, 511 ids), not this one.

**Greedy output was not deterministic before the fix, and the merge did not cause it.** Five identical 2-card runs
gave two different streams: `bd4882...` twice and `09d243...` three times, parting at token 27. Upstream's new
priority stream and lazy CPU lane are inactive here (`g_ls` is created only for one GPU; `g_pool` is null under
EP), and FP16 KV flips the same way. The layer dumps are bit-identical through visit 17 and first differ in visit
18's attention - the first pass > 0, whose message gates call `cublasSgemm` on 512 rows. rocBLAS' SGEMM splits K
with atomic accumulation on gfx906: repeated calls on one matrix differ by 1.5e-8, and two processes' outputs
differ, enough for the FP8 cache's near ties. It reproduces at `c11d62e`, so it predates the merge. The HIP build
now sets `hipblasSetAtomicsMode(..., HIPBLAS_ATOMICS_NOT_ALLOWED)` on its BLAS handles
(`include/strata/hip_compat/cublas_v2.h`); five more identical runs then produced one stream (`09d243...`, 5 of 5).
Decode stays 42.8-44.3 tok/s and 32K 34.39 / 34.43 tok/s, while the 512-id prefill measures 345-354 tok/s (about
2 % below the pre-fix 354-365) because the loop gates then take rocBLAS' deterministic kernels.

**EP is bit-identical to a single GPU in attention, not in the
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
sha256 `7c3b6513...a1d2`). The repaired `/disk/models/mtp.safetensors` is the only complete copy on this machine
(`/disk/models/mtp-fresh/` holds just a download lock, a re-download is still blocked; the corrupt original is
`/disk/models/mtp.safetensors.hole-20261122`), so `--draft` must be pointed at that file and nothing else.

### The verify window: an expert read once, both GPUs, a dense row a token

With the checkpoint whole the proposals continue the text, but the window cost 147 - 197 ms: about a plain token a row,
because `forward()` runs all 48 visits for the window's rows and every row ran the whole decode path. `IK2_WINPROF=1`
(a stream sync a section; `IK2_PROF2=1` gives the attention sections) put a width-8 window at 255 tokens at **209 ms of
experts out of 286** - 5.7 GB of expert bytes, 1.5 GB of them copied into GPU 0 from host memory at ~21 GB/s because a
window's picks (~40 of 128 a layer, 29.6 distinct after grouping) are split over both GPUs' caches by `ep_owner` while
the `T > 1` path only ever looked in GPU 0's - then 59 ms of attention, 6 ms of routing round trips, 4 ms of head and
1 ms of convs.

The window now takes four other paths. They are additive: `T == 1`, the prefill paths and the CUDA paths are untouched.

* **An expert read once for all of its rows.** `nv_limb_mv_g` runs the limb (`__dp4a`) dot that a token's picks use
  (`nv_limb_mv`), a group's expert read once for the up to 8 rows that picked it instead of once a row, its block
  codes expanded once a sub-block for the whole group. The per-row sums are the same lane striding and butterfly, so a
  row's expert output is bit-identical to a token's.
* **Both GPUs for the window.** The window's slots are split by `ep_owner`; each device groups the slots it owns by
  expert on the host (the routing round trip the window makes anyway), looks each up in its own cache (a miss copied
  in on that device's copy stream, the hits' kernels first), runs gate | up | down on its own experts, and GPU 1's
  `y [S][C]` comes back to GPU 0 for the combine (in `y32` past GPU 0's rows). In the steady state the window's misses
  are 0.11 a visit: 25 MB a window over PCIe against 1.5 GB before.
* **A window's dense products.** `mv_f16_multi`: a warp four rows of the matrix, each lane's 8-value chunk of `x`
  shared by them; hipBLAS' GEMM read the same weights at 45 GB/s at 2 .. 8 rows (37 ms of a window's projections
  against 4.3 ms of dense work a token), this at 195. The 8-value order is `mv_f16_token`'s, so a row's sum is
  bit-identical to one token's GEMV.
* **The attention kernel stays per query.** Three blocked designs (one K / V read for the window's queries) were built
  and measured against it, and all lost - at 255 tokens where the read is small and at 32768 where it dominates:

  | design | 255, width 8 | 32768, width 5 |
  |---|---|---|
  | a block a query, four warps splitting its keys (the kernel above) | 55.2 - 58.9 ms a window | **78.1** |
  | the window's queries a warp each, one key stream a warp (256 threads) | 88.6 | 86.0 |
  | a warp a query over the same keys, a block 4 queries x 4 warps a query | - | 92.5 |

  What limits it is not the K / V bytes but the online softmax: each key's update is a chain on (m, s, o), so a warp
  with `nk / P` keys of it serialises, and the per-query kernel's four-way split of one query's keys is what hides the
  load latency. Sharing a row across queries costs shared memory (24.5 KiB a block for four queries) and occupancy, or
  - held in registers - 2 to 4 queries a warp, which measured slower still (`q4`/`ow` per query: 88.6 ms at 255 for
  4, 76.3 for 2). The 32768-token window is therefore still 78 ms of verify; a tensor-core block attention (the CUDA
  path's `attn_block_mma`, compiled out here) or shorter chains (a two-level softmax that keeps four keys in flight)
  is what that needs.
* **The per-layer routing round trip stays**: the expert caches are host-side, so the picks must come to the host.

`IK2_EP_WIN=0` restores the previous window (the scalar row kernels, GPU 0's cache), `IK2_GEMV_WIN=0` the hipBLAS dense
products. Measured with `IK2_DRAFT_ADAPT=0` (every step a window), 63 tokens after the prompt, two cards, FP8 KV:

| prompt | width | tokens a window | drafts accepted | drafting + verifying | tok/s (plain) |
|---|---|---|---|---|---|
| 255 ids | 8 | 2.25 | 37 of 196 (19 %) | 16.9 + 56.7 ms | 30.6 (43.9) |
| 255 ids | 5 | 2.25 | 35 of 112 (31 %) | 15.4 + 42.9 ms | 38.6 (43.9) |
| 255 ids | 3 | 2.10 | 34 of 60 (57 %) | 14.3 + 35.0 ms | 42.6 (43.9) |
| 255 ids, C++ source | 8 | 1.97 | 32 of 224 (14 %) | 16.9 + 54.5 ms | 27.6 (44.4) |
| 255 ids, C++ source | 5 | 2.25 | 36 of 112 (32 %) | 15.3 + 43.5 ms | 38.2 (44.4) |
| 511 ids, README | 8 | 3.32 | 44 of 133 (33 %) | 17.2 + 58.6 ms | 43.8 (42.7) |
| 511 ids, README | 5 | 3.00 | 42 of 84 (50 %) | 15.6 + 45.8 ms | **48.8 (42.7)** |
| 32768 ids | 5 | 2.21 | 24 of 56 (43 %) | 20.9 + 78.0 ms | 22.4 (34.6) |

**Window drift: measured, and not the code.** A later session measured every window configuration 1.4 - 1.5x slower
than the table above, so the previous commit was rebuilt in its own directory (`/tmp/ik2-a`, `git worktree` at
`0fc0094`, same CMake flags and `STRATA_GGML_DIR`; `maya-ik2` md5 `bb801b6b9e8a9a32accedad48fd5426c`) and alternated
with the branch tip's (`a7a9a0f`, md5 `d33c0342fc40fffa43396451521e4edc`) ABBA on the same prompts. The two are the
same binary for this purpose:

| README prompt, `IK2_DRAFT_ADAPT=0`, gen 63 | A `0fc0094` | B `a7a9a0f` |
|---|---|---|
| plain, tok/s | 38.96 | 39.09 |
| width 5: drafting + verifying | 15.6 + 67.5 ms | 15.6 + 67.8 ms |
| width 5, `IK2_WINPROF`: attention / experts / routing / head | 32.15 / 18.13 / 5.79 / 4.77 ms | 32.23 / 18.24 / 5.78 / 4.77 ms |

So there is no regression to bisect: both commits measure the same today, section for section. What is left is the
machine. Its clocks are not it: sampled under load, both cards sit at sclk 1801 MHz (700 idle), mclk 1000 MHz (the
HBM2 maximum), junction 21 - 29 C, GPU 100 % - and the plain path is only 9 % slower today (README plain 38.96
against 42.69 in the earlier session) while the window's *dense products and head* are ~40 % slower (width-8
projections 9.7 -> 13.4 ms) and the expert block is not (22.1 -> 22.9). The box is shared: `load average 7.6` with
other tenants' processes during these runs, and the model now reads from disk at 0.25 GiB/s a load (58 s) where the
earlier session had it page-cached (0.37 GiB/s, 37 s). A window issues ~2400 launches (48 visits) and the T = 8 dense
kernel reads its `x` operand eight times per weight chunk, so it is the most host- and L1/L2-sensitive path in the
engine; that is where a busy machine shows first. Cause unproven beyond that, but the *condition* is: an idle box with the GGUF page-cached.
Record the numbers with the condition or not at all.

Under that shared-box condition the same runs measure (width 5, `IK2_DRAFT_ADAPT=0`, plain in brackets):

| prompt | plain tok/s | drafting + verifying | tok/s | (idle-box row) |
|---|---|---|---|---|
| 255 ids | 43.9 | 15.6 + 65.2 - 66.4 ms (5 runs: 66.4, 65.2, 65.7, 65.3, 64.9) | 26.1 - 26.6 | 15.4 + 42.9, 38.6 |
| 511 ids, README | 39.0 | 15.6 + 67.5 ms | 35.5 | 15.6 + 45.8, 48.8 |
| 32768 ids | 34.6 | 20.9 + 78.1 ms | 22.3 | 20.9 + 78.0, 22.4 |

So under a busy box the spec path is below plain at 255 and 32768 and 0.91x on the README prompt, where the idle-box
numbers were 1.14x; the 32768 window is unchanged (its attention is not what a busy host hurts). A width-8 and
width-3 row were not re-measured here: their idle-box numbers are the table above.

The stream is the target's: the same 63 ids as plain decode on all three prompts except near ties (at 255: 159542 /
152804 tie at 27.0161 / 27.0121; on the README prompt 197020 / 196978 early in the run - a window's summation order
against a token's). At 32768 tokens the window's attention reads 128x the keys it does at 255 while its drafting grows to ~21 ms, so the
window costs 99 ms there and 32K is the one depth where the drafter is well behind plain decoding: the per-query
attention kernel (8 rows, each its own pass over the keys) is what the deep-context window needs next. `IK2_DRAFT_WIDTH=3` is the best width on prose (where a
window accepts 1 - 2 drafts), 5 on the README's shell tail, and 8 never pays (its third draft onwards is mostly
rejected).

A width-8 window at 255 tokens, section by section (`IK2_WINPROF=1`, 8 windows, ms a window):

| section | ms | of it |
|---|---|---|
| setup (the loop's messages, bank copies, taps) | 1.0 | |
| attention block | 23.0 | projections 9.7, gate + o_proj + convs 5.3, the norms 3.5, the attention kernel 3.1, the MoE input 3.1, the K / V convs 1.8, prep + messages 1.4 |
| the routing round trip (its D2H sync) | 4.0 | |
| experts (both GPUs, 6.47 GB a window) | 22.0 | 29.6 distinct a visit, 0.11 copied in, 297 GB/s over the two devices |
| head (8 rows x 200058) | 3.4 | |
| convs + residual | 1.2 | |
| total | 57.9 | (56.7 ms a window measured without the profiler) |

**Determinism of a window.** Two races in the two-device window path were found and fixed: a slot whose lazy copy
(a prompt's or an earlier layer's) was still in flight went with the *hits*, whose kernels do not wait for the copy
stream (it now goes with the misses, as the token path's `slot_sseq` check does), and `g_ep_x / cp / done` were
re-recorded every layer while the other device's wait on them might still be queued (each visit now has its own
events, created on the device whose stream records them). Three repeat runs of the README prompt at width 5 then gave two identical streams and one that flipped a near tie
early; on a cleared machine five consecutive runs at 255 / width 5 / gen 63 (both binaries were idle, no other engine
running) produced **the same 62 ids in all five** (md5 `d15d908bd809bc23ba2af03f89053f86`), so what remains is the
same host-contention sensitivity the window shows everywhere else, not a repeated engine fault.

What is left in the window, in the order the profile puts it: the experts' 297 GB/s is one device's rate (each of the
two devices reads ~3.2 GB in the 22 ms, i.e. 147 GB/s against `nv_limb_mv`'s 294 - the grouped kernel's extra dots are
not free, and the two devices' kernels do not add up); the dense GEMV reads 195 GB/s against `mv_f16_token`'s 840 (a
register-blocked variant and a shared-memory tile of `x` both measured slower and were dropped); the routing's 48 D2H
syncs (4 ms) need device-side expert tables, which this 16 GiB machine cannot hold (`g_ep_split` is off here: GPU 0's
cache holds 1367 slots for the ~1536 experts it owns). A CUDA graph a window size is the other 2 - 3x, and capturing a
window across both devices is still out of scope.

### The window's deep-context attention: the chain is not what limits it

At 32768 tokens the window's verify was 78.1 ms on a busy box and 62.4 - 66.0 on an idle one. Its sections there
(`IK2_WINPROF=1 IK2_PROF2=1`, width 5, 13 windows, ms a window): setup 1.5, **attention block 37.4** - of which the
attention kernel **20.9**, projections 6.8, gate + o_proj + convs 4.1, the norms 3.5, the MoE input 3.1, the K / V
convs 1.5, prep 0.5 - experts 16.4, routing 3.4, head 2.5, convs 1.1; total 65.4. So the kernel is the largest single
section and the only one where a softmax's structure can matter.

The per-key online softmax is a chain: load K -> four-term dot -> a five-step xor butterfly -> (the bias's second
butterfly) -> max -> two `expf` (13 instructions each) -> five FMAs that rescale `(m, s, o)`. `attn_split_gqa4_kernel`
groups **four of a warp's keys** (`j, j+4, j+8, j+12` of the stride-4 set it already had): four K loads together, four
independent dots and butterflies, one max over the four, **one** rescale of `(m, s, o)`, four independent `expf`, and
the value terms summed pairwise; keys left over run through the single-key body, and the split, the four-warp merge and
the `H*P*130` partials are untouched. `IK2_ATTN_2LVL=1` turns it on (default **off**), `IK2_ATTN_2LVL_NK` the first key
count (2048), `IK2_ATTN_KF=2` the two-key group, and `attn_2lvl_mode()` is what `test_attn_two_level` compares. Only a
window's queries (nq >= 2): a token keeps the per-key kernel, whose partials are the scalar path's bit for bit.

It is slower, at both group sizes, and the 32K window is where it should have paid:

| 32768 ids, width 5, `IK2_DRAFT_ADAPT=0`, 13 - 14 steps | verifying | tok/s |
|---|---|---|
| per-key (today's kernel) | **62.4 ms** | 28.6 |
| two keys a group (KF 2) | 64.9 | 27.8 |
| four keys a group (KF 4) | 82.6 | 21.4 |

KF = 4 (the design) loses 32 %: with `G = 3` heads it holds four K rows, four V rows, twelve logits and twelve betas
live at once (~95 - 105 registers) and spills. KF = 2 is a wash (64.9 against 62.4, and the same accepted stream as the
per-key run), which is the informative number: halving the number of rescales per key changes nothing, so at
`nk / P = 256` keys a warp the kernel is not chain-bound - the latency is already hidden by the blocks in flight (the
K / V stream is 65 GB/s, thirteen times below the memory's 890, and adding keys a warp does not move it). The lever
for this kernel is therefore *fewer bytes or fewer instructions a key*, not a shorter chain: the rejected cross-query
sharing designs (86.0 / 92.5 ms) and this one bracket the space, and the block-shaped answer (the CUDA path's
`attn_block_mma`, and a two-level softmax on top of it) needs tensor cores this build does not have.

The four-key kernel and its test were **removed** after the next round of experiments (below): it was off by default,
it lost, and keeping it in step with the per-key kernel's arithmetic cost more than it was worth. Its numbers stand in
the table above.

### Per-key instruction work: measured, and where the bound actually is

The obvious next lever was instructions a key: two `expf` (about 13 each) and two 5-step butterflies per key per head.
Both were built and measured on this branch (`IK2_DRAFT_ADAPT=0`, paired runs, same condition), then reverted:

| 32768 ids | attention kernel, plain decode | 32K window, verify / attention | verdict |
|---|---|---|---|
| baseline (`bccebae`) | 7.46 / 7.47 ms a token | 64.1 / 64.6 ms, 36.0 / 36.2 | - |
| the online softmax in log2 units, `exp2f` (MUFU) for its two exps, and the bias folded into the dot's butterfly | **7.13** (-4.4 %), plain 31.85 -> 32.20 tok/s | not run | **rejected**: the 16640-key FP8 reference check (`attn_decode_fp8 P=128 raw`, 1 of 6144 values) goes from inside its 5e-6 + 2e-4|v| bound to 2.3x outside it (2.8e-5 against 1.2e-5) - a 4 % gain does not buy that |
| the two butterflies merged in natural units (`r` pre-divided by `qk_scale`), `expf` kept | **7.40** (-0.8 %) at 32K, **1.63** against 1.65 at 255 | 63.3, 35.4 (-1.2 %) | **rejected**: noise-level, and it rewrites the plain greedy stream at 255 from the first token (69201... against 145989...) - that prompt's first-token margin is under the fold's ~1e-5 logit shift |

The reference checks and `test_attn_gqa` are what decided it: the fold alone is *correct* to 1e-7 (it passed every
5-check GQA comparison at 0.01 of the 1e-5 tolerance and the 16640-key FP8 check), and `exp2f` is what the FP8
reference catches. So the numbers above are the kernel's *sensitivity to instructions*: removing ~40 % of the per-key
instruction stream (both exps) bought 4.4 %; removing one of the two reductions bought ~1 %. At 32K the kernel is
therefore ~95 % memory latency, not issue: `len = nk / P = 256` keys a block, 64 a warp, and the K / V stream moves at
65 GB/s of the memory's 890. The lever left is memory-level parallelism and load shape per key - the 8-lane x 4-key
mapping (16 B a lane per row instead of 4) and prefetching the next key's row - not arithmetic. A DPP-based 32-lane
reduce (the recon of `mx-llama.cpp`'s `repack-common.cuh`) sits on the same axis as the butterfly that just bought ~1 %,
so it is not where the time is; and the recon's FP8 loader and wide vector attention both measured slower in that fork.

One more observation from the same runs: on the folded binary the 255 window produced four different streams in five
runs (the committed binary had given five identical), one of them with a different accepted-draft histogram. The fold
only shifts logits by ~1e-5, so what it did was *expose* the residual copy race the last commit left open (a slot
evicted by a later layer while an earlier layer's kernel still reads it) - the item is still open, and the committed
arithmetic is the one that repeats.

### The load shape, not the arithmetic: a prefetch that loses and the budget for 8 lanes

The committed kernel's own numbers first: `llvm-readelf -n` on the device object
(`build-hip-gfx906/CMakeFiles/ik2_kernels.dir/src/ik2/ik2_kernels.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx906`) reports
**64 VGPRs, 2 VGPR spills, 48 SGPRs** for the FP8 GQA kernel the plain decode and the window both run. That is the
register budget every change here has to live in - and why KF = 4 (four keys a warp) at ~95 - 105 spilled and lost 32 %.

**A two-deep prefetch of the K row** was the next try, since the dot waits on that 4-byte lane load of an FP8 row: the
next key's row issued at the top of the body (addresses frozen - a ring slot's successor, or a linear cache's next
position clamped to the last key - so the extra load is always a valid row), the body reading the one in flight. It is
bit-identical (`test_attn_gqa` unchanged at 0.06 of its tolerance, 357 checks green) and it is **slower**:

| 32768 ids, plain decode, `IK2_PROF2` over 31 tokens | attention kernel |
|---|---|
| committed kernel | **7.46 / 7.47 ms a token** |
| two-deep K prefetch | **7.63** (+2.3 %) |
| committed kernel again (after reverting) | **7.46** ms, 31.85 tok/s |

So the two loads a key (K and V) were already in flight before the dot consumed K - the compiler had the loop pipelined
inside 64 VGPRs - and the prefetch only added one redundant clamped load a warp and disturbed the schedule. A V
prefetch would add more of the same. The measurement is a clean negative and the kernel is unchanged.

What the kernel *is* bound by is the shape of its reads. Per key a block reads 384 B (the three query heads' slices of
one K head, contiguous) out of a 1 024 B KV row that the other seven K heads' blocks read separately, i.e. a 1 KiB
stride with 37.5 % of each row used by any one block. The numbers agree:

| | K / V bytes | attention kernel | effective |
|---|---|---|---|
| plain decode, 32768 | 1.02 GB a token (15 non-sliding visits x 8 K heads x 32768 keys x 256 B + the sliding layers) | 7.46 ms | **137 GB/s** |
| window, 32768, width 5 | 5.1 GB a window (the same x 5 queries) | 20.9 ms | **244 GB/s** |

Against the memory's 890 GB/s that is 15 % / 27 %; against the ~334 GB/s the 384-B-of-1-KiB pattern can reach it is
41 % / 73 %. The window's better figure is the GQA sharing already working (three heads a row read); the rest is the
stride. Nothing in the instruction stream moves it, which is consistent with the 4.4 % ceiling the exps measured.

**The 8-lane x 4-key mapping** (each 8-lane group owning one key, 16 B a lane of K and V, a 3-step group butterfly)
was specified and *not* built, because its budget does not fit: 8 lanes a key means a lane holds 16 dimensions of the
row, so `q` becomes 16 floats a head (48 for G = 3), the output accumulator `ow` 16 floats a head (48), plus the K and
V rows (16 + 16) and the running `(m, s)` - about **130 VGPRs** against the 64 the kernel has and the ~95 - 105 that
already spilled. The escape (one head a pass, so `q`/`ow` are 16 each) re-reads the K / V rows three times, which is
exactly the sharing that measured slower (86.0 / 92.5 ms). The other escape is a heads-major KV cache, so one block's
read is a whole 1 KiB row - a change to the cache format, the prefill and the scalar path, not to this kernel.

On the acceptance for this attempt: 32K attention <= 6 ms a token is **not** met (7.46), the window is 64.1 ms against
58, and the plain stream is 31.85 tok/s on a shared box (34.4 - 34.6 idle). The prefetch and the mapping are recorded
here as measured negatives; the committed kernel is what runs.

### The planar Q8 row stride: padded, measured, within noise (reverted)

A Q8_0 row of 2304 B is 18 x 128 B, so every row of a planar matrix starts on the same memory channels;
`mx-llama.cpp`'s repack de-aliases by padding a row to `K + 16` bytes (9 - 15 % decode there). Done here for every
planar reader - `upload_planar` (row by row into `N x q8_ld(K)` and `N x q8d_ld(K)`), `mv_q8p_kernel`,
`decode_q8p_f16` (which became a rows x K kernel, three call sites), `mat_rows`, the block path's `tile_rows`, and the
`mv_q8p` test's own planes - with `q8_ld(K) = K + 16` and `q8d_ld(K) = K / 32 + 8` halves (16 B) in the header, and
`kQ8RowPad = 0` on CUDA so its tensor-core tile kernel is untouched. `test_attn_gqa`'s sibling checks stayed green and
`mv_q8p` agreed with its reference at 0.01 of tolerance, so the stride plumbing was right. The 512-id README prompt,
plain, EP, FP8 KV, `gen 63`:

| dense weights | prefill | decode | decode, mean |
|---|---|---|---|
| FP16 (two runs) | 354.4 / 346.0 tok/s | 42.85 / 43.75 tok/s | 43.30 |
| planar Q8, row pad 16 B (two runs) | 355.9 / 356.9 | 43.67 / 43.79 | **43.73** |
| planar Q8, no pad (two runs) | 357.5 / 357.1 | 43.74 / 43.31 | 43.53 |

Two things follow. The de-alias is **+0.5 %** here (43.73 against 43.53), inside this box's +-1.2 % run-to-run spread,
so gfx906's channel interleave is not what limits maya's q8 GEMV - `mv_q8p_kernel` reads a row in 128-B chunks a warp,
four rows a block, and the 16-B shift does not move what it waits on. And planar Q8 is **+1.0 %** over FP16 dense
(43.7 against 43.3), not the 9 - 15 % mx sees on its shapes: the acceptance was `>= 5 %`, so the dense default stays
FP16 and the padding was **reverted** rather than kept as unmeasured plumbing. Prefill is neutral either way at 512
ids (346 - 357 tok/s across all six runs), so the "q8 costs prefill" note in the older table is a chunk/context effect,
not a property of the format.

### Planar NVFP4 slots: the reader audit (design only, nothing built)

The other half of the mx transfer is a planar repack of a resident expert's NVFP4 bytes: quants in one plane
(`[rows][K / 64][32]`), the block scales in another (`[rows][K / 64][4]`, four E4M3 bytes a 64-value block), both rows
16-byte aligned - the slot's size unchanged (36 B of quants plus 4 B of scales per 64 values, exactly today's 36 B
block). The win is the load: today `nv_limb_mv` takes an 8-byte `uint2` of quants plus a scale byte per 16-value
sub-block; planar, two sub-blocks' 16 bytes are contiguous and one `uint4` feeds both, and no load carries a scale.
The audit, before any code:

| reader | reads a slot's bytes as | would need |
|---|---|---|
| `nv_limb_mv` (a token's picks, both phases) | `blob + r * (K/64*36)`, `uint2` + a scale byte | the whole kernel (`K/16` sub-block indexing, both Q8 and NVFP4 branches) |
| `nv_limb_mv_g` (a window's grouped rows) | the same | the same, plus its two loops |
| `row_dot` / `row_dot_2` | a row's 36-B blocks, `block_bytes(type)` | the type dispatch's NVFP4 arm |
| `row_dot_multi` / `row_dot_pair_multi` | the same, several inputs | the same, twice |
| `experts_gate_up_kernel`, `experts_gate_up_swiglu_kernel`, `experts_down_kernel` | `row_dot_pair_multi`'s helpers | (through the helpers, but each passes `eb[]` offsets) |
| `experts_gate_up(_swiglu)_g_kernel`, `experts_down_g_kernel` | the grouped limbs (`36 * (k/4)` indexing) | both grouped limbs |
| `nv_step_load` + `grouped_expert_mma` | 36-B blocks a lane (CUDA only) | (unused on HIP; would have to match if the host layout changed for both) |
| the CPU lane's `post_expert` | `Ly.host[r] + e * stride[r]` | the host pools |
| the fill path: `read_into_strided` (load) and `ExpertCache::copy_in` (3 memcpys a slot) | `Ly.eb[r]` bytes | the host layout only, if the repack happens once at load |

That is **seven distinct decode sites** (three scalar helpers, three grouped/scalar kernel families, the limb pair),
plus the CPU lane and the fill path: more than the four the brief allows, and the *prefill* path (the `experts_*_g`
kernels and `row_dot_multi`) is exactly the part that would have to be rewritten - the prefill is the path that
already regressed once when its kernels were touched. So this stops at the design, as instructed, and the prototype it
calls for is the next step and its own turn: one planar `nv_limb_mv` (NVFP4 only) behind a switch, fed by a device-side
repack of one slot into a scratch buffer (no second copy in the real path - the load-time host repack would make the
planar layout *the* layout), and a kernel-only GB/s figure against the 294 GB/s the linear limb kernel measures today
(the 0.1217 ms an 8-expert layer-visit at 48 visits). The migration list above is the scope of that change.

### ... and the prototype says no: the layout axis is closed

A self-contained `/tmp` TU (the repo's `NvInput`, `pack_limbs`, `nv_expand`/`nv_dot_vals`/`nv_dot_words`,
`nv_limb_dot`, `ue4m3`, `wsum` copied verbatim, `-ffp-contract=off`, the same grid, threads and 128-thread blocks as
`expert_limb_mv`'s launches) ran the linear kernel and a planar one on the same data: 8 experts, C = 2304, F = 1152, a
slot of 4 478 976 B (1 492 992 a part), 200 launches a measurement. The planar kernel reads two 16-value sub-blocks from
one 16-byte quant load and the pair's two scales from one 16-bit load, the scales in a plane of their own
(`[rows][K/64][4]`), the quants in `[rows][K/64][32]`, rows 16-byte aligned; the repack is a device kernel (one thread a
64-value block) run outside the timed loop.

| 8 experts a launch | linear | planar | |
|---|---|---|---|
| gate / up | 341.7 GB/s, 0.0699 ms | 284.9 GB/s, 0.0838 ms | 0.83x |
| down | 254.2 GB/s, 0.0470 ms | 215.7 GB/s, 0.0554 ms | 0.85x |
| a visit (both) | **306.6 GB/s, 0.1169 ms** | **257.1 GB/s, 0.1393 ms** | **0.84x** |

The baseline reproduces the kernel this tree runs (the doc's 294 GB/s / 0.1217 ms a visit; this harness is 4 % faster,
the same shape). The reason the planar form loses is in the metadata: `llvm-readelf -n` on the object gives **41 VGPRs**
(`Gate`), 38 (down), no spills for the linear kernel against **64 VGPRs**, no spills, for the planar one - 128-thread
blocks, so 12 blocks a CU against 8. The two sub-blocks in flight cost ~23 registers (two `uint4` quants and two packed
`NvInput`s) and a third of the occupancy, and this kernel lives on occupancy: it is 84 % of nothing but latency. A
one-sub-block planar kernel would keep the VGPRs and do exactly today's loads (8 B of quants + a scale byte), only from
two planes - the scale plane being one more stream, i.e. no win to be had there.

Once at fill the repack costs **0.191 ms an expert** (23.4 GB/s over 4.5 MB) - off the hot path, and a cold fill of 1536
experts would add ~0.3 s to the 39 s the pinned copy already takes; not the objection. The objection is the kernel. The
planar down phase agrees with the linear kernel to 6.5e-4 relative worst (a reordering over 72 - 144 terms), so the
maths is comparable; it just cannot be made faster this way.

**So the layout axis is closed**: 257 GB/s a visit against 307 today, well under the 380 GB/s the migration would have
needed to be worth its seven decode sites. The limb kernels keep their 36-byte blocks; the remaining limb headroom is
the SwiGLU epilogue and the memory floor the doc already records, not the slot's layout.

### The window's ordering, audited, and two fixes it turned up

A code-level pass over the two-device window path (per-visit events `x / cp[d] / done`, the single-buffer pack a
device, the hit / miss classification, the peer writes) finds **no live race**, and the reason is worth stating because
it is load-bearing: every layer's routing ends in a full-stream synchronisation of GPU 0, and GPU 0 has by then waited
on `done[l-1]`, which GPU 1 records after its pack copy and its y-rows copy. So at the top of layer *l* both devices
have finished layer *l-1* - the packs can be rewritten, `Ed.get` can evict, and the previous layer's kernels cannot
still be reading what it frees. The events are indexed by visit and re-recorded only in the next window, whose
predecessor ended synchronised; a slot whose lazy copy might be in flight (a prompt's or an earlier layer's) is
classified with the misses and waited for on the copy stream before its kernels run. The residual risks are recorded
in open work: the single-buffer packs and the eviction ordering both *depend* on that per-layer sync.

The audit did turn up two real faults in the configuration the fit check now enables, both fixed here:

* **`--draft` aborted** as soon as the dense weights were planar Q8: the drafter's base logits go through
  `k::q8_tile`, a tensor-core kernel this build does not have, and the fit check had started selecting planar Q8 on a
  free box. The HIP build now leaves the drafter on the FP16 head copy the fold already makes (0.9 GiB on GPU 0)
  instead of calling into the missing kernel.
* **A drafter now keeps the dense weights FP16** (`draft_path.empty()` in the fit check): with planar Q8 the window's
  projections at T > 1 lose ~30 % (39.2 against 48.1 tok/s on the 512-id README prompt, verify 67.6 against 45.7 ms).
  Plain decode keeps planar Q8's +1 % when no drafter is loaded.

With those two, the spec path runs in its intended configuration and the tests stay at 357 checks, 0 failures.

### The window's determinism: measured, then fixed

Plain decode was 5 of 5 identical (md5 `5608e98aa117`, 512-id README prompt, 64 tokens, 42.73 - 43.51 tok/s, FP16
dense) on the final tip; **the window was 11 of 12** - the exception (`win1`, the first of twelve) diverged at
generation token 52 (model position 563, 47528 where the other eleven pick 76), its accept counts differed at that
window (4 accepted drafts against 5), and the top-2 margin at that position was **above** the 0.05 the near-tie
print uses: not a tie flip, a different logits vector, roughly 1 in 12 windows. `IK2_EP_WIN=0` was 3 of 3 identical
but runs at 21.0 - 21.4 tok/s (no workaround).

The single-buffer packs, the lazy-copy classification (`sseq > g_seq_done`, the ring's aliasing sync) and the
per-visit event chain were each re-read under this evidence and are conservative; what was not covered is the hit
path's assumption itself: a *hit* takes its slot's bytes as final without any wait, and the copies that fill a slot
do not all join the sseq protocol that the test reads (the window's own copies are inline `copy_in`s, and the
prefill's prefetched ones are ordered by `pf_evs`, which only the prefill waits on). The fix is the conservative
one: **every row of a window layer waits on the layer's copy-stream event before any expert kernel runs** - the
hit / miss split is gone, and with it the assumption. Measured after (2026-11-22, 19:50 - 20:40, load ~2 - 3, 29 -
33 C): **12 of 12 identical to plain** (`5608e98aa117` twelve times, 47.89 - 49.02 tok/s, mean 48.17, 3.00 tokens
a window, verify 45.5 - 47.0 ms), 48.17 against 48.88 tok/s before (-1.5 %), `ik2_kernels_test` 357 checks, 0
failures, plain unchanged (44.04 tok/s, `5608e98aa117`, same session, its path untouched) and the prefill 419 - 429
tok/s with a warm page cache (344 on a cold one right after the tests - its path is untouched either way). Twelve clean runs alone are not conclusive at a 1-in-12 rate (p ~ 0.35); the change is
justified as well by being strictly conservative - it adds an ordering and removes none.

### The serve-side window (opt-in, `IK2_SERVE_DRAFT=1`)

`--serve` with a drafter now has the CLI's fallback window: each step proposes `B - 1` drafts, verifies the anchor and
the drafts in one `forward()` (T = B <= 8), accepts the leading run of drafts that equals the target's own pick at
that row, rolls the convs back to the kept tokens, and carries the target's next token.  Greedy: the argmax, exactly
the CLI's `dr[n] == argmax` rule, and **with the width pinned (`IK2_DRAFT_ADAPT=0 IK2_DRAFT_WIDTH=5`) the ids are the
plain stream's, 64 of 64, twice** (q8, 262144, EP, DRAFT_CTX 65536).  Sampled: each row's token is the request's own
draw through the serve loop's sampler (`pick`: temperature / top_k / top_p over the top 1000 - the same distribution
plain sampling uses), so an accepted draft is a target sample and the sequence distribution is the plain one; the
draws differ in count, so equal ids are only guaranteed at temperature 0.  The width comes from the same `SpecCtl`
the resident path uses, now with two depth-aware fixes: a width not yet measured at this context is priced from its
255-depth prior scaled by the plain token's measured time, and windows that did not pay make the next probe wait
longer (up to 8x).  With `IK2_DRAFT_ADAPT` on, the width sequence follows the measured timings, so the stream may
differ from plain at ulp-level ties only (observed at 2 positions on one prompt; the CLI has the same property).

**It does not pay in serve at 262144, so it is opt-in**, measured on this tip at a 255-token prompt (63-token
answers, `IK2_DRAFT_CTX=65536`, EP, FP8 KV, 262144 context):

| dense | drafter | tok/s | mix |
|---|---|---|---|
| FP16 | off (plain) | **41.0** | - |
| FP16 | window, adaptive | 33.6 | 16 windows (15 of 39 drafts), 32 plain |
| q8 | off (plain) | **45.6** | - |
| q8 | window, adaptive | 30.9 | 11 windows (9 of 22 drafts), 43 plain |

A window's step costs ~+20 ms (FP16) or ~+60 ms (q8) more than a plain token in the serve path, so even a 38 %
acceptance cannot pay it; the controller backs off to plain but the probes still cost ~25 %.

**The serve loop is not the overhead.** `IK2_SERVE_PROF=1` times a step's sections (propose 15.3 ms, the window's
`forward` 52 - 96 ms, sampling / rollback / emit under 0.1 ms), and the *CLI* at the same settings is identical:
`--max-context 262144`, FP16, width 5, 255-token prompt -> `drafting 15.3 ms, verifying 57.0 ms`, 2.25 tokens a step,
**31.1 tok/s** against the serve path's 31.7.  What the window pays for is the **expert cache at 262144**: the same
63-token generation misses **901** experts there against **90** at 131072 (the window's union is ~40 experts a layer
against the plain path's 8), i.e. ~1 GiB of expert copies, ~0.25 s of the 2.0 s.  So the window pays where the union
is resident - 131072: 48.0 against 43.1 tok/s, 3.0 tokens a step - and cannot at 262144 with the current
`--cache-gb 14` (K/V 4.2 GiB leaves too little for the union), which is why it stays behind `IK2_SERVE_DRAFT=1`.  Streaming, cancellation and SIGTERM
were re-checked with it on: one delta a token, a client disconnect ends the request (`cancel=True`), and the engine
exits in ~1 s.

### The event sweep's two other victims, and the serve check

The commit that introduced the window's per-visit copy events (`a7a9a0f`) renamed them in one sweep and took two
branches with it: the tensor-parallel path and the single-token EP path ended up waiting on `g_win.cp[d][vis]`, an
event only the draft-window branch records - so `--serve` with a drafter loaded (which decodes at T = 1, because
`spec_on` needs `g_fast`, set only on the single-GPU path) died on its first token with `invalid resource handle`,
and then hung in `exit()`. Restored: each of those branches waits on `g_ep_cp[d]`, the event it records a few lines
above. The audit of every wait against its branch's record: tensor-parallel `g_ep_cp[d]` / `g_ep_done`; the window
`g_win.cp[d][vis]` / `g_win.done[vis]`; the single-token EP `g_ep_cp[d]` / `g_ep_done`; the canonical-hits path
`g_ep_done`. Verified through the real server (2 cards, `--draft /disk/models/mtp.safetensors`, `IK2_EP=1 IK2_KV=fp8
IK2_DENSE=q8 IK2_DRAFT_CTX=65536`, `--max-context 262144 --cache-gb 14`): two non-streaming requests and one
streaming one, 41 - 46 tok/s on the answers (plain decode: through `--serve` the drafter is resident but idle on
this build, since `spec_on = drafter && g_fast && g_block` and `g_fast` is only set on the single-GPU path - wiring
the window into the serve loop is still open work), clean stop; and the CLI window 12 of 12 identical again after
the change (`5608e98aa117` twelve times, 47.90 - 48.99 tok/s, tests 357 checks 0 failures).

## Disabled on gfx906

* **Tensor-core attention and experts** (`mma.sync m16n8k16`, Hopper `wgmma`, `cp.async`, `ldmatrix`,
  `q8_tile`, the grouped-block kernels): PTX, compiled out under `STRATA_USE_HIP`; their launchers abort with
  `maya-ik2: <fn>: tensor-core path not built for HIP/gfx906`. The GQA split attention
  (`attn_split_gqa_kernel`), the scalar split attention it falls back to, and the limb (`__dp4a`) expert kernels
  run instead. `IK2_FLASH=1`, `IK2_ATTN_MMA=1`, `IK2_BLOCK=1`, `IK2_EXPERT_MMA=1`
  are refused at startup rather than run.
* **The block path** (a verify window's attention, `attn_block_mma`) and CUDA graphs: off (`IK2_GRAPH=0`).
* **The drafter through the general prompt path** (`--draft`): correct with EP (2.0 - 3.5 tokens a window) and its
  verify window now runs the grouped limb experts over both GPUs' caches and `mv_f16_multi`, which took a width-8
  window at 255 tokens from 286 ms (profile: 209 of it experts) to 58 ms. It still pays only where the drafter is
  accepted: the README's shell tail gives 48.8 tok/s against 42.7 plain at width 5, while prose and C++ source are at
  or below plain decoding (255 ids: 42.6 tok/s at width 3 against 43.9). Off by default, and `--serve` has no spec
  path for it (its `spec_step` is the block path's, which this build does not have).
* **Vision**: the mmproj file exists but the vision encoder is not built for HIP here.

## Cold start

The 14.7 GiB of experts are read once into pinned host memory, then each GPU's cache is filled with the
experts it owns. On this machine the read runs at ~0.4 GiB/s when the page cache is cold (37 s) and ~2 s when
warm; the per-GPU fill is 0.6 s; `READY` follows. Keep the GGUF page-cached for serving sessions.

## Open work

1. **The attention kernel's read shape** (the one axis still open): 7.46 ms a token at 32K for 1.02 GB of K / V, i.e.
   137 GB/s, and 244 GB/s in a window. A block reads 384 B - its three query heads' slices - out of a 1 KiB K / V
   row that seven other blocks read separately, so the pattern's own ceiling is ~334 GB/s and the kernel sits at
   41 % of that (the window, 73 %). The redesign is a whole-row read: one block taking all eight K heads' slices for
   its keys (24 query heads of accumulators - 3 KiB of LDS a block - or a heads-major cache, which is a format change
   for the cache, the prefill and the scalar path). Everything cheaper on this kernel is closed and measured: the
   instruction stream (4.4 %), a two-deep prefetch (-2.3 %), 8-lane x 4-key (over the register budget), cross-query
   sharing (0.84 - 0.86x), log2/exp2f (2.3x over the FP8 reference check).
2. **Cross-device graphs**: the token loop's pipeline gap is still the biggest single loss - the attention phase
   measures ~13.8 ms a token against ~6 ms of kernel service in isolation, `attn_res_rms_f16` alone costs ~25 µs a
   call with an idle queue (11.7 µs pipelined), and the same ~10 µs gap sits around every small kernel. Capturing a
   *window* would take most of it, but the capture crosses two devices; per-layer, two-device graphs are the untested
   alternative.
3. **The window's pack-ring constraint** (the determinism defect found here on 2026-11-22 is fixed - see the
   window section): the window path keeps one pinned 5 632-byte group table a device and rewrites it every layer,
   while the copy that reads it is asynchronous. The ordering argument that makes it safe is the per-layer routing
   round trip: layer *l*'s full-stream sync on GPU 0 comes after GPU 0's wait on `done[l-1]` (recorded on GPU 1
   after its pack copy), so the host cannot rewrite a pack before its previous reader has finished; the same sync is
   what orders `Ed.get`'s evictions against the previous layer's kernels on both devices. The events are per visit
   and only re-recorded in the next window (each window ends synchronised). Any change that removes that per-layer
   sync (the device-side routing this document asks for elsewhere) must give the packs a ring and wait on the copy's
   own event.
4. **The limb kernels' epilogues and floor**: the fused gate / up SwiGLU epilogue is 21 % of a visit (0.025 ms) and
   MUFU-latency bound, and the limb kernels read at 294 GB/s against the memory's 890 (an 8-expert visit 0.1217 ms).
   The planar slot repack measured 0.84x (its own section), so what is left here is the epilogue and the read floor,
   not the layout.
5. `IK2_DENSE=q8` remains opt-in and is now never chosen when a drafter is loaded: it is +1 % on plain decode at 512
   ids (43.7 against 43.3) and the window's projections at T > 1 lose ~30 % with it (39.2 against 48.1 tok/s), which
   is what the fit check's `draft_path.empty()` clause enforces.
