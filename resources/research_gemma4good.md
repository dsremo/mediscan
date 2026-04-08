# Gemma 4 Good Hackathon — Research Output
## Source: Gemini Deep Research — 2026-04-08

## 1. Gemma 4 Models

| Model | Params | Context | Modalities | Best For |
|---|---|---|---|---|
| E2B | ~2.3B | 128K | Text, Image, Audio | Phones, IoT, Raspberry Pi |
| E4B | ~4.5B | 128K | Text, Image, Audio | Complex reasoning, function calling |
| 26B MoE | 26B (3.8B active) | 256K | Text, Image, Video | Laptops/workstations |
| 31B Dense | ~30.7B | 256K | Text, Image, Video | Maximum quality |

- License: Apache 2.0
- Native function calling + structured JSON + system prompts
- Audio support on E2B/E4B (unique in this size class)
- 31B scored 89.2% on AIME 2026, 3rd among all open models on Arena AI

## 2. Edge Deployment Speeds

| Device | Model | Speed |
|---|---|---|
| Snapdragon 8 Gen 3 (Pixel 9, S24) | E2B | 20-35 tok/s |
| Snapdragon 8 Gen 3 | E4B | 12-20 tok/s |
| iPhone 15 Pro (MLX) | E2B/E4B | 40+ tok/s |
| Raspberry Pi 5 | E2B | 7.6 tok/s generation, 133 tok/s prefill |

Deployment: MediaPipe LLM Inference API or llama.cpp (GGUF)
Quantization: Q4_K_M reduces 31B with only 1.5-2% accuracy drop

## 3. Past Winners — Gemma 3n Impact Challenge (600+ entries)

- **1st: Gemma Vision** — AI assistant for developer's blind brother. Chest-mounted phone + Bluetooth gamepad. Personal story = huge factor.
- **2nd: Vite Vere Offline** — Cognitive disabilities, complete offline
- **3rd: 3VA** — AAC technology

**Winning traits:** Personal story, offline-first, hardware integration, MediaPipe/flutter_gemma, explainable reasoning

## 4. Healthcare Datasets

| Dataset | Images | Focus | Source |
|---|---|---|---|
| PASSION | 4,901 | Pigmented skin from Sub-Saharan Africa (pediatric) | Open |
| eSkinHealth | 5,623 | Neglected Tropical Skin Diseases (West Africa) | Open |
| SIIM-ISIC 2020 | 33,126 | Melanoma dermoscopic | Open |
| SLICE-3D | 400,000+ | Skin lesions from 3D photography | Open |

## 5. Key Repos

- `unslothai/unsloth` — Optimized GGUF quants + fine-tuning
- `google-ai-edge/LiteRT` — TFLite conversion
- `google-gemini/gemma-cookbook` — Official mobile deployment guide
- `ggerganov/llama.cpp` — GGUF inference on Pi/Android
- Google AI Edge Gallery app — Testing

## 6. Competition Strategy

- Most teams will build text-only chatbots or "chat with PDF" — boring
- Healthcare + developing countries = underexplored, high-impact
- $50K winner = full-stack product + compelling narrative + offline + multimodal + function calling
- $5K winner = solid notebook, lacks polish/story
- Use PASSION/eSkinHealth datasets = shows rigor other teams miss

## 7. Winning Formula

1. Multimodal: camera for skin/wounds/eyes + audio for symptom description
2. Native function calling: first-aid instructions, medication reminders, local health resources
3. Offline-first: MediaPipe or llama.cpp, E2B/E4B quantized to 4-bit
4. Narrative: specific user persona (community health worker in rural Kenya)
5. Demo video: story-driven, not dry technical walkthrough
