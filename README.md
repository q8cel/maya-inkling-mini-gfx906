<h1 align="center">Inkling 2 Mini on ONE RTX 3090: 128 tok/s plain, up to 212 with DSpark</h1>

<p align="center"><b>Thinking Machines' 26B looped MoE, all of it in 24 GB of VRAM. 126 to 128 tok/s plain decode, 182 to 207 tok/s through the HTTP server on new code and up to 212 tok/s on edits with the official DSpark drafter, 2,742 tok/s prefill, up to 344K context, images and tools. llama.cpp gets 57 tok/s on the same card. Also runs on 8 GB GPUs at ~41 tok/s, Linux and Windows, NVIDIA CUDA and AMD HIP, with continuous batching (<code>--parallel</code>) and host-RAM conversation caching (<code>IK2_SLOT_KEEP</code>).</b></p>

<p align="center">
<a href="docs/INKLING2-MINI.md"><b>Read the guide</b></a> ·
<a href="#quick-start"><b>Quick start</b></a> ·
<a href="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF"><b>Weights on Hugging Face</b></a>
</p>

| Text, greedy (1× RTX 3090) | Plain decode, tok/s | With DSpark drafter, tok/s | llama.cpp b12590 |
|---|---:|---:|---:|
| Prose | 126 | 131 | 57.0 |
| New code | 125 | 149 (HTTP server: 182 to 207) | |
| Editing a document | 124 | 212 | |
| Copying code | 124 | 188 | |
| A 64K token document | 104 | 125 | 42.7 (65,536 deep) |

One RTX 3090 at a 300 W limit, Unsloth `UD-Q4_K_XL`, FP8 KV cache, a cold prompt every time. Short-prompt plain decode hits 128.7 tok/s CLI (129.5 tok/s over HTTP). Prefill runs at 2,742 tok/s on 4K tokens, 2,361 on 64K and 2,079 on 128K. Every launch and log is in the [guide](docs/INKLING2-MINI.md#numbers).

## What you get

* **`maya-ik2`**, an engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): the 24 layers, the loop (layers 6 to 17 run 3 times, 48 layer visits per token), the relative bias, the short convs and the attention residuals.
* **The whole model stays in 24 GB VRAM.** All 3,072 experts (14.7 GiB), the dense weights and an FP8 KV cache fit on one 24 GB card: 344,064 tokens without the drafter, 262,144 with it. Nothing is read from RAM or disk while it answers.
* **Runs on 8 GB GPUs too (~41 tok/s).** [`ik2-mini-8gb.json`](docs/inkling2-mini/ik2-mini-8gb.json) keeps the experts in pinned host RAM with a VRAM cache and computes cache misses (both NVFP4 and Q8_0 layers) on the CPU without stalling the GPU. Measured 40 to 43 tok/s at 16K context on an RTX 5070 Laptop 8 GB (vs 12.5 tok/s in llama.cpp on the same laptop).
* **Speculative decoding with the official DSpark drafter.** Thinking Machines ships a 3-layer DSpark drafter (`mtp.safetensors`) with the model. The engine verifies 2 to 8 tokens at a time in one CUDA-graphed pass and drafts only when that pays: 182 to 207 tok/s through the server on new code, up to 212 tok/s on edits, while prose stays as fast as plain decode. Greedy and sampled output stay exact.
* **Continuous batching (`--parallel N`) and conversation caching (`IK2_SLOT_KEEP`).** Pass `--parallel N` (2 to 8, single GPU) to serve concurrent requests over a shared paged global KV pool (~205 tok/s aggregate for 4 concurrent 300-token prompts on one 3090; DSpark still kicks in when 1 slot is decoding). On the default `--parallel 1` path, `IK2_SLOT_KEEP` (default `8`) parks displaced conversations (KV, conv histories, sliding windows, and drafter state) in host RAM so interleaved agent calls restore a 54.5K-token chat in ~70 ms instead of re-prefilling for 12+ seconds.
* **The original weights.** The experts of layers 0 to 19 are Thinking Machines' own NVFP4 bits, the way Unsloth's GGUF carries them. A 3090 has no FP4 hardware, so the engine decodes the 4-bit codes in registers and feeds the tensor cores.
* **Linux, Windows (MSVC / CUDA 13), and AMD (HIP).** Native Windows builds work out of the box (including `--parallel` serving and 64-bit safetensors offsets), and AMD GPUs build via ROCm/HIP (`-DSTRATA_ENABLE_HIP=ON`; the gfx906 port runs `--parallel 1`).
* **Images, tools, reasoning control, streaming** through OpenAI and Anthropic compatible APIs on `/v1`.

## Quick start

Linux or Windows (MSVC), an NVIDIA GPU (`sm_80` or newer, CUDA 12.8+ / 13.x; 24 GB for full resident VRAM or 8 GB+ with `ik2-mini-8gb.json`) or an AMD GPU with ROCm/HIP, CMake 3.21+, Ninja, and Python 3.10+.

```bash
git clone https://github.com/alesha-pro/maya-inkling-mini && cd maya-inkling-mini
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub

# the engine (NVIDIA CUDA; set CMAKE_CUDA_ARCHITECTURES to your GPU, e.g. 86, 89, 120)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-ik2

# the image encoder (skip it for text only)
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision -j 24 --target maya-vision

# the weights (17.6 GB), the drafter (0.6 GB) and the tokenizer files
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir models
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check

python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini.json --host 127.0.0.1 --port 18000
```

For an **AMD GPU (HIP)**, configure with `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON` and your `AMDGPU_TARGETS` / `CMAKE_HIP_ARCHITECTURES` (for example `gfx1201`; the gfx906 port's measurements and settings are in [docs/AMD_IK2_MI50.md](docs/AMD_IK2_MI50.md)). See the [guide](docs/INKLING2-MINI.md#quick-start) for Windows MSVC and AMD HIP build flags.

Then talk to `http://127.0.0.1:18000/v1` with any OpenAI or Anthropic client, model `inkling-2-mini`. There are 5 configs in [`docs/inkling2-mini/`](docs/inkling2-mini/): the default (`ik2-mini.json`: DSpark drafter, 262K context, CPU vision), longest context (`ik2-mini-long.json`: 344K, no drafter), text only (`ik2-mini-text.json`), GPU vision (`ik2-mini-gpu-vision.json`), and 8 GB GPUs (`ik2-mini-8gb.json`: 16K context, ~41 tok/s). Add `"--parallel", "4"` to `args` in a config to serve concurrent requests.

## Why it's faster than llama.cpp

llama.cpp runs about 7,350 kernel launches per token for this model. A token here is 757 launches replayed as one CUDA graph, and the GPU never waits for the host between the embedding and the logits.

After that a token costs what its weights weigh: 3.7 GB read from VRAM, half of it experts. So the kernels are built around memory speed. Attention reads the KV cache at up to 846 GB/s, the NVFP4 experts go through the tensor cores at 745 GB/s, and an FP8 cache beats FP16 deep in a context because it is half the bytes.

The drafter's window has its own path: the tokens' expert picks are grouped on the GPU so an expert is read once for all the tokens that picked it, and the window runs as one graph too. 5 tokens cost 15.6 ms against 7.8 ms for one.

The [guide](docs/INKLING2-MINI.md#how-it-works) has the details, and what I tried that didn't work.

## Quality

The engine computes the model the GGUF holds, no pruning and no requantizing. On 96 GSM8K questions, greedy: 74 solved with plain decode and 77 with the drafter (the drafter only changes which pass computes a token, so the difference is noise). Against the float64 reference in `ref/inkling2.py` the engine picks the same top token on 45 to 46 of 47 positions, and the reference agrees with llama.cpp on 47 of 47. Batched `--parallel 4` decode matches solo `forward()` with median KL ~3e-8 and 0 argmax disagreements across 125 checked rows (`IK2_SLOTS_CHECK=32`).

## Limits

* `--parallel N` (2 to 8 concurrent requests) is single-GPU and non-managed KV only; DSpark speculates when 1 slot is active and switches to batched decode when 2+ run together. Two GPUs run as expert parallel (`IK2_EP=1`) or tensor-parallel experts (`IK2_TP=1`) with `--parallel 1`, and the drafter runs on one GPU. `IK2_SLOT_KEEP` currently applies to `--parallel 1`.
* It reads `UD-Q4_K_XL` only. `UD-Q8_K_XL` keeps the last 4 layers' experts in BF16 and the engine doesn't load them.
* Decode slows down with depth, because 8 of the 48 layer visits attend over the full prefix: 16 KB of cache per token of context is read for every new token.
* The drafter helps code, edits and structured text. On free prose it accepts about 2 tokens per window and the engine mostly decodes plain.
* The image encoder runs on the CPU in the default configs. A new image takes about 4.5 s on a 48-core EPYC, the same image again 0.5 s.
* Tested hardware: Linux on RTX 3090 (with `gpu-unlock`, Ubuntu 22.04, driver 610.43) and AMD RX 9070 XT (`gfx1201`, ROCm 10.2.0 via HIP); Windows 11 (MSVC + CUDA 13.5) on RTX 4090, RTX 5090, RTX 5070 Laptop 8 GB, and dual RTX 5080 + RTX 4060 Ti. On stock upstream `main`, the AMD HIP build uses the compatibility / emulated matrix-core path; this branch carries the native scalar HIP/gfx906 port below.
* Community HIP/gfx906 port on 2 × MI50 (expert parallel only, scalar attention and limb (`__dp4a`) expert kernels, no tensor cores): 44.2 tok/s decode at a 255-token context and 414–442 tok/s prefill against llama.cpp's 30.4 / 359–389 there; the drafter and the image encoder are off. [docs/AMD_IK2_MI50.md](docs/AMD_IK2_MI50.md).

## About this repo

This is [Project Maya](https://github.com/mw00/project-maya) v1.1.70, the engine family by Peasant Smith ([@mw00](https://github.com/mw00)) and the Maya contributors, which grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. `maya-ik2` started as the Inkling 2 Small engine from Maya's pull requests [#981](https://github.com/mw00/project-maya/pull/981) and [#1079](https://github.com/mw00/project-maya/pull/1079). This branch adds the loop, the path that keeps every expert in VRAM, the tensor core kernels for attention and experts, FP8 KV for prefill, GPU sampling candidates, speculative decoding with the official drafter, and the Mini profile of the server.

Community contributions merged here:
* [@cobra91](https://github.com/cobra91): initial Windows / CUDA 13 port and two-GPU expert parallel.
* [@PrasoonPratham](https://github.com/PrasoonPratham): Windows MSVC hardening (`SIGUSR1`, 64-bit offsets, `PrefetchVirtualMemory`, `VirtualAlloc`, CMake UTF-8 & Threads, [#3](https://github.com/alesha-pro/maya-inkling-mini/pull/3)).
* [@ImagineNotChetng](https://github.com/ImagineNotChetng): 8 GB GPU cache + CPU miss path at ~41 tok/s and `ik2-mini-8gb.json` ([#4](https://github.com/alesha-pro/maya-inkling-mini/pull/4)).
* [@joeldeteves](https://github.com/joeldeteves): native `--parallel` continuous batching with paged global KV ([#5](https://github.com/alesha-pro/maya-inkling-mini/pull/5)).
* [@DeseretSaint](https://github.com/DeseretSaint): AMD HIP build and compatibility layer for `maya-ik2` ([#7](https://github.com/alesha-pro/maya-inkling-mini/pull/7)).
* [@mkultraware](https://github.com/mkultraware): Windows `serve_multi` build fix ([#10](https://github.com/alesha-pro/maya-inkling-mini/pull/10)) and host-RAM conversation caching (`IK2_SLOT_KEEP`, [#11](https://github.com/alesha-pro/maya-inkling-mini/pull/11)).

Most of the core work is offered upstream in [mw00/project-maya#1095](https://github.com/mw00/project-maya/pull/1095). The rest of Maya (GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-flash and the app around them) is still in the tree but isn't maintained here: use [upstream Maya](https://github.com/mw00/project-maya) for those.

Weights: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its drafter by Thinking Machines, GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).

## License

The code is MIT, as upstream ([LICENSE](LICENSE)). The model weights are Apache 2.0.
