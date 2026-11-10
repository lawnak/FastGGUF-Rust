fastgguf in Rust: run GGUF language models on your own computer, on the CPU and on NVIDIA, AMD or Intel GPUs.

### What's new

{{CHANGES}}

### Download

**Windows x64:** `fastgguf-rs-{{TAG}}-windows-x64.zip`. Unzip and run; no installation. One `fastgguf.exe` works on any 64-bit Intel / AMD CPU and picks the fastest kernels yours supports. It does not need the Visual C++ Redistributable.

```
fastgguf.exe chat model.gguf      # chat in the terminal
fastgguf.exe serve model.gguf     # web UI at http://localhost:8080 + OpenAI-compatible API
fastgguf.exe selftest             # check every kernel on this PC
```

Linux / macOS: `cargo build --release` (see the README).

### Features

- **Models:** Gemma 2/3/3n/4, Qwen2/3, Qwen3 MoE, Llama and Mistral
- **Quant types:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS
- **CPU:** AVX2, AVX-VNNI and AVX-512 VNNI kernels in one binary, chosen at start-up; int8 GEMM for reading prompts in batches
- **GPU:** OpenCL on NVIDIA, AMD and Intel, loaded at run time. Each GPU kernel is checked against the CPU before use, and models larger than VRAM are split between GPU and CPU
- **Server:** web UI and OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming). Up to 4 models at once with parallel / chain / loop modes, prompt-cache reuse and an optional API key
- **TurboQuant KV cache** (`FASTGGUF_KV=turbo`): 5–9× less memory for long conversations
- **Unicode everywhere:** Arabic, CJK and emoji in the terminal, the web UI and file paths

### Speed

On an Intel i5-1135G7 (8 threads) with `gemma-3-1b-it`, reading the prompt runs at 52–67 tokens/s and generation at 21–25 tokens/s, as fast as or faster than the C version.

**Full changelog:** {{COMPARE}}
