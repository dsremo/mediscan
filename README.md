# MediScan — AI-Powered Offline Medical Triage Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org)
[![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%20E4B-4285F4?logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Healthcare in your pocket — no internet required.

MediScan is an AI-powered medical triage assistant that helps community health workers in underserved areas assess skin conditions and provide preliminary guidance. Built with Google Gemma 4 for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).

## The problem

- **3.5 billion people** lack access to essential health services (WHO)
- Sub-Saharan Africa has **less than 1 dermatologist per million people**
- 80% of pediatric populations in these regions have **untreated skin conditions**
- Existing health apps require internet — useless in the exact rural areas that need them most

## The solution

MediScan puts triage support directly into the hands of community health workers:

1. **Take a photo** of a skin condition
2. **Describe symptoms** (text or voice)
3. **Get instant triage** — urgency level, first aid, when to seek care
4. **All offline** — runs entirely on-device, no network needed

## Features

- **Multimodal analysis** — Gemma 4 processes both image and text in a single forward pass
- **Structured triage output** — four discrete urgency levels (low / moderate / high / emergency) instead of free-text, so downstream UI can route correctly
- **Function calling** — first-aid instructions, medication reminders, clinic finder routed through tool calls instead of hallucinated text
- **Offline-first** — runs on a mid-range Android phone via Gemma 4 E4B (4.5B params, 4-bit quantized, ~3 GB on disk)
- **Privacy by construction** — no patient data ever leaves the device, so HIPAA / DPDP / GDPR are non-questions
- **22 skin conditions** — fine-tuned on a CC0 dermatology dataset

## Architecture

```
+---------------------+
| Camera / mic / text |
+----------+----------+
           |
           v
+---------------------+
| Gemma 4 E4B (LoRA)  |       fine-tuned with Unsloth on a single Kaggle T4
| 4-bit GGUF          |       no quantization-aware retraining needed
| ~3 GB on disk       |
+----------+----------+
           |
           v
+---------------------+       +-----------------------+
| Triage output       | ----> | Function calls:       |
| { urgency, summary, |       |  - first_aid(cond)    |
|   first_aid }       |       |  - medication_reminder|
+----------+----------+       |  - find_nearest_clinic|
           |                  +-----------------------+
           v
+---------------------+
| Android UI          |       runtime: Ollama / llama.cpp / MediaPipe
+---------------------+
```

## Tech stack

| Component | Technology |
|---|---|
| Base model | Google Gemma 4 E4B (4.5B params, quantized to 4-bit) |
| Fine-tuning | Unsloth + LoRA, ~1 hour on a free Kaggle T4 GPU |
| Image input | Gemma 4's native vision tower (no separate ViT pipeline) |
| Function calling | Native Gemma 4 tool-use format |
| Web demo | Gradio (deployable to HuggingFace Spaces) |
| Training data | [Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) — 22 classes, 1.4 GB, CC0 license |
| Edge deployment | Ollama / llama.cpp / MediaPipe |

## Project structure

```
mediscan/
├── src/
│   ├── mediscan_core.py      # Core inference pipeline + function calling
│   └── app.py                # Gradio web demo
├── notebooks/
│   └── mediscan_finetune.py  # Unsloth fine-tuning on Kaggle GPU
├── requirements.txt
└── README.md
```

## Quick start

```bash
git clone https://github.com/dsremo/mediscan.git
cd mediscan
pip install -r requirements.txt
python src/app.py            # launches Gradio demo on http://localhost:7860
```

For edge deployment to a phone, see `notebooks/mediscan_finetune.py` for the GGUF export step.

## Key design decisions

**Why Gemma 4 instead of a dermatology-specific CNN?** A dedicated CNN (EfficientNet, ResNet) would give better single-image classification accuracy on the 22 trained classes, but it can't accept text symptom descriptions, can't generate structured triage output, and can't route through function calls. A multimodal LLM trades a few accuracy points for an end-to-end pipeline that fits in one model — which matters when you're shipping 3 GB to an offline phone, not 3 GB plus a separate CNN plus a separate NLP head plus glue code.

**Why E4B (4.5B) rather than the smaller E2B variant?** E4B is the smallest Gemma 4 variant that retains the vision tower at full fidelity. Going smaller forces the user to ship a separate vision encoder, which negates the "one model, offline" pitch.

**Why fine-tune with LoRA instead of full fine-tuning?** Full fine-tuning a 4.5B model needs >=40 GB GPU RAM. LoRA adapters fit in a single 16 GB T4 (Kaggle's free tier), the adapter is ~25 MB on disk so it can be shipped separately from the base model, and the inference penalty is negligible after merge.

**Why function calling instead of letting the model just describe what to do?** Three reasons. First, LLMs hallucinate medication doses; routing through a `first_aid(condition)` function call ensures the dosage table comes from a verified static source. Second, "find nearest clinic" needs actual GPS / contacts integration that the model cannot itself perform. Third, structured outputs are auditable — a triage flagged "high urgency, seek care within 4 hours" can be logged and reviewed; "you should probably see someone soon" cannot.

**Why offline-first, not "cloud with offline fallback"?** The target user — a community health worker in a rural clinic — has intermittent connectivity at best. A "cloud with fallback" design quietly degrades to fallback-only most of the time, but is sold as cloud-capable, which makes both modes worse. Building offline-first forces the inference loop to be fast enough on a phone, period.

## Known limitations

- **Triage, not diagnosis.** MediScan flags urgency and suggests first aid; it does not name diseases with medical confidence and is not a substitute for a clinician. The 22-class dataset covers common dermatological presentations but is not exhaustive.
- **Skin tone bias.** Dermatology datasets historically over-represent lighter skin. The Skin Disease Dataset has more diversity than ImageNet-derived alternatives but is not balanced across Fitzpatrick scale skin types. Performance on darker skin types is likely degraded — a known limitation, not a fix-in-this-repo problem.
- **Phone hardware constraints.** Gemma 4 E4B at 4-bit needs ~4 GB free RAM. Sub-4-GB-RAM phones either OOM at load time or run unbearably slowly. The target device class is a 2022+ mid-range Android phone.
- **No validation against clinician ground truth.** The training set has labels but there is no held-out set with dermatologist-confirmed triage labels. Accuracy is measured against the dataset's classification labels, not against real triage outcomes.
- **English only at present.** The model handles English well; multilingual triage (Hindi, Swahili, Yoruba) requires further fine-tuning data that wasn't available in the hackathon window.

## What I would do differently with more time

- Validate on a clinician-labelled triage set rather than the dataset's classification labels.
- Run a fairness audit across Fitzpatrick types and publish per-subgroup recall numbers.
- Replace function calling for `find_nearest_clinic` with an actual offline geo-lookup (OpenStreetMap clinic data, pre-bundled per country).
- Add a "I don't know" path. Right now the model always returns a triage; sometimes the right answer is to abstain and say "this needs a person to look at it."
- Multilingual triage with a small per-language LoRA layered on the base.

## Acknowledgments

- **Google DeepMind** — Gemma 4 model family
- **Skin Disease Dataset** — CC0 public domain dermatology images
- **Unsloth** — efficient fine-tuning framework that made the Kaggle-T4 budget feasible

## License

Apache 2.0.

---

Built for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).
