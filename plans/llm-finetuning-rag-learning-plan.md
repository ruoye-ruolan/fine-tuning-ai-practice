# Fine-Tuning, Retrieval-Augmented Generation & Deployment

*cs.LG · self-study curriculum · rev 1.2 — sep 2026*  
A 16-week practitioner's plan · three tracks · eight checkpoints · ≈ 8–10 h / week

**Abstract.** This curriculum takes you from a working understanding of transformers to three production skills: adapting open-weight models with supervised and preference-based fine-tuning (**Track A**), grounding any model in your own data with retrieval-augmented generation (**Track B**), and turning open-weight models into fast, observable endpoints you can afford to run (**Track C**). Sixteen weeks — thirteen for the tracks, three for the capstone — and eight checkpoints, every one of them a shipped artifact, because the model you trained and the retriever you measured are the only proof of learning that counts.

**Contents:** [§1 Tracks](#1-three-tracks-one-system) · [§2 How to run it](#2-how-to-run-this-plan) · [§3 Checkpoints 00–07](#3-the-checkpoints) · [§4 Exit criteria](#4-exit-criteria) · [§5 Beyond](#5-beyond-the-curriculum) · [§6 References](#6-references)

## Timeline

| Weeks | Checkpoint | Track | You ship |
|---|---|---|---|
| 0–1 | ckpt-00 | shared | Foundations notebook |
| 1–3 | ckpt-01 | A | Golden set, then an SFT run scored against it |
| 3–5 | ckpt-02 | A | QLoRA on a 7–9B model → merged GGUF → served in Ollama |
| 5–7 | ckpt-03 | A | DPO pass + eval report (base vs. SFT vs. DPO) |
| 7–9 | ckpt-04 | B | From-scratch RAG CLI, then the LlamaIndex rebuild |
| 9–11 | ckpt-05 | B | Hybrid search + reranker + eval harness with a before/after table |
| 11–13 | ckpt-06 | C | Two deployments (local GGUF, serverless vLLM/SGLang) + load-test table |
| 13–16 | ckpt-07 | A + B + C | Capstone domain assistant: tuned model, measured retriever, benchmarked stack |

> Each checkpoint is a project you ship, not a chapter you finish. Focus hands off A → B at week 7 and B → C at week 11. Deployment skill starts accruing back at ckpt-02 — the first time you serve your own weights — then compounds in ckpt-06. All three tracks leave from a single origin and meet again at a single point: the capstone (ckpt-07), which runs from week 13 to week 16 — three weeks, because integration is where the earlier checkpoints' hidden bugs surface.

## §1 Three tracks, one system

### Track A · weeks 0–7 — Fine-tuning

Change what a model *is*: its format, style, domain fluency, and behavior. You will move from full supervised fine-tuning on small models to QLoRA on 7–9B models to preference tuning with DPO — always on hardware you can actually get.

### Track B · weeks 7–11 — Retrieval-augmented generation

Change what a model *knows* at answer time: fresh, private, or long-tail facts fetched from your own corpus. You will build the pipeline by hand first — embeddings, chunking, vector search — then make it trustworthy with hybrid retrieval, reranking, and evals.

### Track C · weeks 11–13 — Deployment

Change what a model *costs to run*: the same weights can stream in 200 ms or crawl, depending on the engine. You will go from Ollama on your laptop to vLLM/SGLang endpoints with continuous batching and quantized serving — and a latency/cost table you measured yourself.

> **Why all three:** in production they split the work — fine-tuning teaches the model *how to behave*, retrieval supplies *what is true right now*, and deployment decides whether anyone can afford to use it. The capstone wires your tuned model into your measured retriever on a serving stack you benchmarked — exactly the shape of most real LLM systems.

## §2 How to run this plan

- **Entry requirements.** Python, Git, Docker, and Linux fluency — on Windows, that means WSL2. If any of these is shaky, it will tax you more than any LLM concept in this plan; patch it with [MIT's Missing Semester](https://missing.csail.mit.edu) before week 0.
- **Cadence.** Three sessions a week: one for reading and theory, two at the keyboard. Roughly 8–10 hours total; the week ranges below stretch gracefully if you have less.
- **The ship rule.** Never end a checkpoint without a runnable artifact in a Git repo — a training script, a merged model, an eval report. If nothing shipped, the checkpoint isn't done.
- **Keep a lab notebook.** One markdown file per checkpoint: what you ran, hyperparameters, what broke, what you'd change. This becomes your portfolio narrative for free.
- **Pin your versions.** Freeze a `requirements.txt` the day you start each project. This stack moves fast, and half of all "bugs" are silent library upgrades.
- **Refresh the model pick.** The checkpoints below name specific models; they are the September 2026 defaults, not permanent ones. The open-weight lineup turns over every few months, so re-check the pick the day each checkpoint starts — and prefer a family with an Apache-2.0 license and both base and instruct checkpoints.

| Hardware you have | What it handles in this plan |
|---|---|
| No GPU (laptop only) | All of Track B — RAG runs fine on CPU with API or local quantized models — plus the llama.cpp / Ollama half of Track C, and every reading week. |
| Colab free / Kaggle (T4 16 GB) | ckpt-01 SFT on 0.5–2B models; ckpt-02 QLoRA on 7–9B models at short context. Enough for the whole plan, with patience. |
| Colab Pro (L4 / A100) | Comfortable QLoRA and DPO on 7–9B, longer context, faster iterations. |
| Local 16 GB (RTX 5080, via WSL2) | All of Track A: ckpt-01 full SFT on ≤2B with an 8-bit optimizer; ckpt-02 QLoRA and ckpt-03 DPO on 7–9B at 2–4k context. Track C locally at INT4 (AWQ/GPTQ) and FP8 in vLLM; the unquantized bf16 baseline of an 8–9B model (≈16–18 GB of weights alone) is the one thing that goes to a rented GPU. See the Blackwell note below. |
| Local 24 GB (3090 / 4090) | Everything here, offline, including serving your merged model with vLLM. |
| Rented (RunPod, Lambda) | Burst capacity for ckpt-03 preference runs, the bf16 baseline in ckpt-06's table, and the serverless deploy — a few dollars per session. |

> **Blackwell note (RTX 50-series, compute capability 12.0).** Do every CUDA step inside WSL2 Ubuntu: vLLM and SGLang are Linux-only, Unsloth is supported on Linux and WSL, and bitsandbytes' Windows wheels are limited. Install a current PyTorch release with `sm_120` kernels (early `cu128` builds had Blackwell cuBLAS bugs — don't pin an old one) and a bitsandbytes built for the same CUDA major version; mixing a `cu13x` torch with a `cu12` bitsandbytes is the most common breakage. If Triton or `torch.compile` misbehaves, disable compilation and use SDPA attention, and clear Unsloth's compiled cache after any torch upgrade. The lowest-friction route is a pinned NVIDIA NGC PyTorch container in Docker (WSL2 backend) — it also doubles as your ckpt-06 serving environment.

## §3 The checkpoints

### ckpt-00 · shared · weeks 0–1 — Foundations refresh

*Make sure the transformer isn't a black box before you start bending it.*

- Transformer anatomy — tokens, embeddings, attention, the KV cache — at the depth needed to reason about memory use and context limits, not to derive the math.
- The adaptation ladder: prompting → RAG → fine-tuning, and the question that routes every project: *is the gap in the model's knowledge, or in its behavior?*
- The Hugging Face stack: `transformers`, `datasets`, `tokenizers`; loading a small model and watching what the tokenizer actually does to your text.

**Ship →** A notebook that loads a small open model (Qwen3.5 0.8B or Gemma 4 E2B), inspects its tokenizer on tricky inputs, generates with three sampling settings, and logs VRAM use for each. Already fluent? Pass this in one sitting and move on.

**Materials**

- **watch** · [Karpathy — "Let's build GPT: from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the best two hours on what a transformer actually computes
- **watch** · [3Blue1Brown — Attention in transformers](https://www.3blue1brown.com/lessons/attention) — visual intuition before any code
- **read** · [Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the canonical diagram walkthrough
- **work** · [Hugging Face LLM Course, ch. 1–2](https://huggingface.co/learn/llm-course) — the transformers / datasets stack, hands-on
- **work** · [Anthropic — Prompt engineering guide](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) — the first rung of the ladder; exhaust it before any heavier tool
- **do** · [Tiktokenizer](https://tiktokenizer.vercel.app) — paste tricky strings and watch how they tokenize
- **skim** · [Transformers quicktour](https://huggingface.co/docs/transformers/quicktour) — pipeline, AutoModel, generate()

### ckpt-01 · track A · weeks 1–3 — Supervised fine-tuning

*Take a base model and teach it to follow instructions in your format.*

- Evals before training: write a 30–50 prompt golden set for *your* task — with reference answers or a short rubric — before you touch a trainer. Every Track A model (SFT, QLoRA, DPO) is scored against this same set, so the comparisons in ckpt-02 and ckpt-03 mean something.
- Full fine-tuning mechanics: cross-entropy loss on next tokens, learning-rate schedules, epochs vs. overfitting, and catastrophic forgetting — what it looks like and how to notice it early.
- Chat templates and dataset formats (Alpaca, ShareGPT); masking prompts so you train only on completions; why 1,000 clean examples beat 50,000 scraped ones — and how synthetic pipelines (generate → filter → refine) produce them at scale.
- TRL's `SFTTrainer` end to end: packing, effective batch size via gradient accumulation, checkpoint saving and resuming.

**Ship →** Your golden set, committed before the first training run. Then an SFT run on a 0.6–2B *base* checkpoint (Qwen3 1.7B-Base is the safe default; pick a family that ships pre-trained weights, since teaching a base model to follow instructions is the point) over 1–5k instruction pairs (a Dolly-15k subset, or pairs you wrote for a task you care about). Score base vs. tuned on the golden set and record the diff in your lab notebook.

**Materials**

- **work** · [HF LLM Course, ch. 3 — fine-tuning](https://huggingface.co/learn/llm-course/chapter3/1) — the training loop end to end
- **read** · [TRL SFTTrainer guide](https://huggingface.co/docs/trl/sft_trainer) — packing, completion-only loss, the knobs you'll actually turn
- **read** · [Chat templates](https://huggingface.co/docs/transformers/chat_templating) — where silent formatting bugs live
- **use** · [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — clean instruction pairs to subset for the ship goal
- **skim** · [Self-Instruct paper](https://arxiv.org/abs/2212.10560) — where the synthetic-instruction recipe started
- **read** · [Raschka — Build an LLM (From Scratch), ch. 6–7](https://www.manning.com/books/build-a-large-language-model-from-scratch) — with runnable [companion code](https://github.com/rasbt/LLMs-from-scratch)

### ckpt-02 · track A · weeks 3–5 — PEFT: LoRA, QLoRA & quantization

*Fine-tune 7–9B models on one GPU by touching less than 1% of the weights.*

- The VRAM ledger — weights + gradients + optimizer states + activations — and why full fine-tuning of a 7B model wants a cluster while QLoRA fits on a free Colab T4.
- LoRA's moving parts: rank `r`, `alpha`, dropout, target modules; what merging adapters back into the base actually does.
- QLoRA specifically: 4-bit NF4 storage, double quantization, paged optimizers — training through a frozen quantized base.
- The inference-quantization landscape as a separate concern: GGUF / llama.cpp for local serving, GPTQ and AWQ for GPUs; plus the shortcut tools worth knowing — Unsloth, Axolotl, LLaMA-Factory.
- The 2026 wrinkle: the current small families (Qwen3.5, Gemma 4) are hybrid architectures — Gated DeltaNet layers in Qwen3.5, mixed local/global attention with unusual head sizes in Gemma 4. Confirm that your trainer, PEFT's target modules, and llama.cpp's GGUF converter support the exact checkpoint *before* you commit; when in doubt, fall back to a plain transformer (Qwen3 8B, Llama 3.1 8B), which has the most tutorials.
- Licensing before publishing: Qwen3.5 and Gemma 4 are Apache-2.0; Llama ships custom terms with naming and attribution rules for derivatives. Check before you push a merged model anywhere public.

**Ship →** A QLoRA fine-tune of a 7–9B model (Qwen3.5 9B is the September 2026 default; Qwen3 8B or Llama 3.1 8B if you want the beaten path) on a domain dataset. Merge the adapter, export to GGUF, and serve it locally through Ollama so someone else could pull and run your model — carrying the chat template and stop tokens into the Modelfile, then re-running your ckpt-01 tricky prompts to prove the served model behaves like the trained one; a training/serving template mismatch is the classic silent failure. Quietly, this is also your first rep of Track C.

**Materials**

- **read** · [LoRA paper](https://arxiv.org/abs/2106.09685) — §1–4 suffice; the idea fits on a napkin
- **read** · [QLoRA paper](https://arxiv.org/abs/2305.14314) — focus on NF4 and paged optimizers
- **work** · [Unsloth docs & notebooks](https://docs.unsloth.ai) — free-Colab-sized QLoRA runs
- **read** · [Raschka — Practical tips for finetuning with LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — rank and alpha choices, backed by experiments
- **skim** · [PEFT LoRA guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — the API you'll import
- **do** · [Ollama](https://github.com/ollama/ollama) — import your merged GGUF and serve it

### ckpt-03 · track A · weeks 5–7 — Preference tuning & evaluation

*Align the model's behavior with preferences — then prove the fine-tune actually helped.*

- RLHF at concept level — reward models, PPO — and where practice went next: direct methods (DPO) when you have preference pairs, and RL with verifiable rewards when the task has a checkable answer.
- DPO and its siblings (ORPO, KTO): preference pairs, the `beta` knob, the role of the reference model.
- GRPO and verifiable rewards: the DeepSeek-R1 recipe — sample several completions, score them with a programmatic check (unit tests, exact answers, format compliance), reinforce the better ones. Runnable on one GPU with TRL's `GRPOTrainer` or Unsloth's GRPO notebooks; the design question is whether *your* task has a reward you can compute, not whether the method is fashionable.
- Evaluation beyond loss curves: `lm-evaluation-harness` for standard benchmarks; LLM-as-judge and its biases (position, verbosity); and your ckpt-01 golden set, which matters more than any leaderboard.

**Ship →** A DPO pass on top of your ckpt-02 model using a preference dataset (an UltraFeedback subset works). Then an eval report: base vs. SFT (ckpt-02) vs. DPO on the golden set you wrote in ckpt-01, with a judge prompt, including the cases where DPO made things worse. Optional but recommended: one short GRPO run on a task with a programmatic reward, to feel the difference.

**Materials**

- **read** · [DPO paper](https://arxiv.org/abs/2305.18290) — the trick: your language model is secretly a reward model
- **work** · [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) — beta, reference models, the training loop
- **skim** · [HF alignment-handbook](https://github.com/huggingface/alignment-handbook) — maintained SFT → DPO recipes to adapt (the older "Llama 2 with DPO" post uses a retired TRL API)
- **skim** · [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) — GRPO and verifiable rewards; read for the recipe, not the scale
- **work** · [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) — reward functions as plain Python; a single-GPU run on a checkable task
- **use** · [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — ready-made preference pairs
- **read** · [Judging LLM-as-a-Judge (MT-Bench)](https://arxiv.org/abs/2306.05685) — the judge biases you must design around
- **work** · [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — run two or three benchmarks pre/post

### ckpt-04 · track B · weeks 7–9 — RAG from first principles

*Build retrieval before you install a framework — every component by hand, once.*

- Whether you need a retriever at all: for a corpus that fits in a few hundred thousand tokens, long context plus prompt caching often beats RAG on both quality and simplicity. RAG earns its keep when the corpus is large, changes often, or needs per-query access control — decide this on paper before writing the pipeline.
- Ingestion before intelligence: parsing real-world PDFs, HTML, and tables into clean text — the unglamorous step where production RAG quality is won or lost.
- Embeddings: bi-encoders, cosine similarity, the MTEB leaderboard as a menu; running `sentence-transformers` models locally.
- Chunking — fixed, recursive, structure-aware, semantic; overlap — and why chunking choices move answer quality more than the choice of vector database.
- Vector stores and approximate search: FAISS, Chroma, Qdrant, pgvector; HNSW in one diagram; when brute force is honestly fine.
- The naive pipeline — ingest → chunk → embed → top-k → stuff context → generate with citations — and its classic failure modes: wrong chunks, lost context, confident answers with no support.

**Ship →** A from-scratch RAG CLI over your own PDFs or notes: `sentence-transformers` + FAISS + any chat model (your ckpt-02 model in Ollama closes the loop). Then rebuild the same tool in LlamaIndex and write down what the framework was hiding from you.

**Materials**

- **skim** · [Lewis et al. — the original RAG paper](https://arxiv.org/abs/2005.11401) — for framing; the field moved, the shape didn't
- **work** · [sentence-transformers quickstart](https://sbert.net) — embed and compare in ten lines; pick your model off the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **work** · [LangChain — RAG from scratch](https://github.com/langchain-ai/rag-from-scratch) — short videos and notebooks matching this checkpoint's ethos
- **read** · [Pinecone — Chunking strategies](https://www.pinecone.io/learn/chunking-strategies/) — the decision that moves your numbers most
- **skim** · [Docling](https://github.com/docling-project/docling) — layout-aware parsing for the PDFs and tables you will actually meet
- **skim** · [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — index types, and when flat brute force is honestly fine
- **work** · [LlamaIndex starter tutorial](https://docs.llamaindex.ai) — for the framework rebuild

### ckpt-05 · track B · weeks 9–11 — Advanced retrieval & RAG evals

*Take retrieval from "demo" to "trustworthy" — and measure the difference.*

- Hybrid search: BM25 + dense vectors fused with reciprocal rank fusion; metadata filtering as the cheapest big win.
- Rerankers as a second stage: cross-encoders (`bge-reranker`, Cohere Rerank) reordering a wide candidate set.
- Query transforms — multi-query, HyDE, decomposition — plus small-to-big / parent-document retrieval and contextual retrieval for chunks that need their surroundings.
- RAG evaluation: faithfulness, answer relevancy, context precision and recall in the RAGAS style; building a 30–50 question golden set; a sober look at when GraphRAG or agentic retrieval earns its complexity.

**Ship →** Your ckpt-04 system upgraded with hybrid search + a reranker, an eval harness over your golden set, and a before/after table quantifying the lift. If a fancy technique doesn't move your numbers, the notebook says so.

**Materials**

- **read** · [Anthropic — Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — with failure-rate numbers to beat
- **read** · [Weaviate — Hybrid search explained](https://weaviate.io/blog/hybrid-search-explained) — BM25 + dense + reciprocal rank fusion
- **use** · [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — a drop-in cross-encoder second stage
- **read** · [HyDE paper](https://arxiv.org/abs/2212.10496) — hypothetical documents for hard queries
- **work** · [RAGAS](https://docs.ragas.io) — faithfulness and context metrics over your golden set
- **skim** · [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) — know when graphs earn their complexity

### ckpt-06 · track C · weeks 11–13 — Serving & deployment

*Turn model files into a fast, cheap, observable endpoint.*

- The serving landscape and when each engine fits: `llama.cpp` / Ollama for laptops and edge boxes; vLLM and SGLang for production GPUs — PagedAttention, RadixAttention, continuous batching; TensorRT-LLM when you're squeezing the last drop out of NVIDIA silicon.
- The four numbers of inference — time-to-first-token, inter-token latency, throughput, cost per million tokens — and how batch size trades them against each other.
- Speed levers: prefix caching, speculative decoding, chunked prefill, tensor parallelism across GPUs; quantized-serving tradeoffs (GGUF vs. AWQ/GPTQ vs. FP8) chosen per target hardware.
- Packaging: an OpenAI-compatible endpoint with streaming, inside Docker with the GPU runtime; serverless GPU platforms (Modal, RunPod) vs. renting a box; structured output via grammar-constrained decoding.
- Ops before users arrive: load testing your endpoint, autoscaling and cold starts, rate limits and basic guardrails.

**Ship →** Deploy your ckpt-03 model twice. Local: GGUF through llama.cpp / Ollama. Production: vLLM or SGLang behind a Dockerized, OpenAI-compatible streaming endpoint on a serverless GPU platform. Load-test both and publish the table — TTFT, tokens/s, and $/1M tokens at three quantization levels: on a 16 GB card that means INT4 (AWQ/GPTQ) and FP8 locally, with the unquantized bf16 baseline measured on the rented or serverless GPU.

**Materials**

- **work** · [vLLM docs](https://docs.vllm.ai) — quickstart plus the OpenAI-compatible server; your production default
- **read** · [Databricks — LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — TTFT, inter-token latency, and batching, with numbers
- **work** · [SGLang docs](https://docs.sglang.ai) — RadixAttention; benchmark it against vLLM on your workload
- **skim** · [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF quant levels for the local deploy
- **work** · [Modal docs](https://modal.com/docs) — the serverless GPU path for the ship goal
- **skim** · [Outlines](https://github.com/dottxt-ai/outlines) — grammar-constrained structured output

### ckpt-07 · track A + track B + track C · weeks 13–16 — Capstone: a domain assistant

*Wire all three tracks into one deployed system — tuned voice, retrieved knowledge, measured serving.*

- Division of labor in real systems: fine-tune for format, style, and tool use; retrieve for facts that change; why baking knowledge into weights is usually the wrong call.
- Choosing the ckpt-06 serving recipe for *this* workload — latency budget, expected concurrency, GPU spend — written down before you deploy.
- Why this gets three weeks: integration is where the earlier checkpoints' hidden bugs surface — a template mismatch between training and serving, a retriever tuned on a different chunker, an eval harness that never ran against the served model. Budget for it instead of discovering it.
- Production hygiene: response caching, guardrails — including prompt-injection defenses for anything your retriever feeds the model — monitoring for drift, and a plan for re-indexing when the corpus updates.

**Ship →** The capstone: a domain assistant for a corpus you care about — your ckpt-03 model on your ckpt-06 serving stack, your ckpt-05 retrieval pipeline behind a FastAPI endpoint, the eval suite in CI, and a one-page report covering quality *and* latency/cost. This repo is the portfolio piece.

**Materials**

- **read** · [Huyen — AI Engineering](https://huyenchip.com/books/) — especially the evaluation and inference chapters
- **read** · [Yan — Patterns for LLM Systems & Products](https://eugeneyan.com/writing/llm-patterns/) — evals, RAG, and guardrails in one map
- **work** · [FastAPI](https://fastapi.tiangolo.com) — streaming responses for your endpoint

## §4 Exit criteria

### Track A — fine-tuning

- [ ] I can sketch the VRAM ledger for full FT vs. LoRA vs. QLoRA from memory, and pick the right method for a given GPU.
- [ ] A merged GGUF model I trained is published — under a license that allows it — where someone else can pull and run it.
- [ ] My DPO model beats my SFT model on the golden set I wrote before training — and I can say by how much, and where it regressed.

### Track B — RAG

- [ ] My retriever's context precision and faithfulness are measured numbers, not vibes.
- [ ] I can defend my chunking strategy for a specific corpus and describe two alternatives I rejected.
- [ ] Hybrid search + reranking beats naive top-k on my eval set, and I have the table to show it.

### Track C — deployment

- [ ] I can quote my model's TTFT, tokens/s, and $/1M tokens at three quantization levels — from my own load test, not a blog post.
- [ ] I can explain why continuous batching and prefix caching raise throughput, and when speculative decoding actually helps.
- [ ] A Dockerized, OpenAI-compatible endpoint serving a model I fine-tuned is live where a friend could hit it with an API key.

### Convergence

- [ ] One deployed endpoint combines my tuned model, my retriever, and a serving stack I chose deliberately — with an eval report a stranger could reproduce.

## §5 Beyond the curriculum

Six areas that sit outside the 16-week clock but will pull on you within your first real project. None needs a dedicated checkpoint — fold each one in the moment it becomes blocking.

### Prompt & context engineering

The cheapest lever, and the rung of the adaptation ladder you exhaust before every heavier tool in this plan.

*Start — [Prompt Engineering Guide](https://www.promptingguide.ai)*

### Agents & tool use

Function calling, multi-step loops, and MCP — where agentic RAG (ckpt-05) and tool-use fine-tuning (ckpt-07) both lead.

*Start — [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) · [MCP](https://modelcontextprotocol.io)*

### Security & prompt injection

Your retriever ingests documents, and documents can carry adversarial instructions — the top practical vulnerability in RAG and agent systems.

*Start — [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [Willison's series](https://simonwillison.net/series/prompt-injection/)*

### Observability & LLMOps

ckpt-03's eval discipline made permanent: tracing, logging, and regression evals running continuously against production traffic.

*Start — [Langfuse docs](https://langfuse.com/docs)*

### Synthetic data generation

How most fine-tuning datasets are actually made now: strong models generating, filtering, and refining the pairs Track A trains on.

*Start — [distilabel](https://github.com/argilla-io/distilabel)*

### Multimodal inputs

The 2026 small models take images natively, and some take audio. A domain assistant over PDFs with figures or scanned pages will want that — fold it in when your corpus stops being text-only.

*Start — [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Docling](https://github.com/docling-project/docling)*

## §6 References

1. [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — A. Karpathy. Video series; "Let's build GPT" is the core session.
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — J. Alammar. The canonical visual walkthrough.
3. [Hugging Face LLM Course](https://huggingface.co/learn/llm-course). Free; covers the transformers/datasets stack and fine-tuning.
4. [Attention in Transformers](https://www.3blue1brown.com/lessons/attention) — 3Blue1Brown. Visual intuition for attention.
5. [TRL documentation](https://huggingface.co/docs/trl). SFTTrainer, DPOTrainer, and GRPOTrainer references.
6. [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) — S. Raschka. Best single book for training mechanics.
7. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021.
8. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023.
9. [Unsloth documentation](https://docs.unsloth.ai). Fast QLoRA notebooks that fit free Colab.
10. [PEFT documentation](https://huggingface.co/docs/peft). Adapter configs, merging, target modules.
11. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023.
12. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI. Standard benchmark runner.
13. [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020.
14. [sentence-transformers documentation](https://sbert.net). Embedding and cross-encoder models.
15. [LlamaIndex documentation](https://docs.llamaindex.ai). Framework reference plus advanced retrieval guides.
16. [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Choosing an embedding model.
17. [RAGAS documentation](https://docs.ragas.io). Faithfulness, relevancy, context precision/recall.
18. [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic. Prepending chunk context before embedding.
19. [vLLM documentation](https://docs.vllm.ai). PagedAttention, serving your merged model.
20. [AI Engineering](https://huyenchip.com/books/) — C. Huyen. Production framing for all three tracks.
21. [SGLang documentation](https://docs.sglang.ai). RadixAttention, structured output, high-throughput serving.
22. [llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org. GGUF quantization and CPU/edge inference.
23. [LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — Databricks. TTFT, inter-token latency, and batching tradeoffs, with numbers.
24. [Modal documentation](https://modal.com/docs). Serverless GPU deployment patterns.
25. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025. GRPO and RL with verifiable rewards.
26. [The Alignment Handbook](https://github.com/huggingface/alignment-handbook) — Hugging Face. Maintained SFT and DPO training recipes.
