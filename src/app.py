"""
MediScan — Gradio Web Demo
Runs on HuggingFace Spaces (free) or locally.

For the live demo submission, this will be deployed to:
https://huggingface.co/spaces/dsremo/mediscan
"""

import gradio as gr
from PIL import Image
from typing import Optional
import json

# Try to import the core module
try:
    from mediscan_core import (
        SYSTEM_PROMPT, MEDISCAN_TOOLS,
        parse_triage_response, format_triage_for_display,
        UrgencyLevel, TriageResult
    )
    from safety import escalate_for_safety
except ImportError:
    from src.mediscan_core import (
        SYSTEM_PROMPT, MEDISCAN_TOOLS,
        parse_triage_response, format_triage_for_display,
        UrgencyLevel, TriageResult
    )
    from src.safety import escalate_for_safety


# Global model reference (loaded once)
_model = None
_processor = None


def load_model():
    """Load model — tries Kaggle GPU first, then CPU fallback."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "google/gemma-4-E4B"

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        # CPU fallback (slower but works on HF Spaces free tier)
        # Use smaller E2B model for CPU
        model_id = "google/gemma-4-E2B"
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype="auto",
        )

    _processor = AutoProcessor.from_pretrained(model_id)
    return _model, _processor


def analyze_condition(image: Optional[Image.Image], symptoms: str, history: list):
    """Main analysis function called by Gradio."""
    if not symptoms and image is None:
        return history + [
            {"role": "assistant", "content": "Please provide a photo of the condition or describe your symptoms."}
        ], history

    model, processor = load_model()

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for msg in history:
        messages.append(msg)

    # Current message
    if image is not None:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": symptoms or "Please analyze this skin condition."}
        ]
    else:
        user_content = symptoms

    messages.append({"role": "user", "content": user_content})

    # Generate
    import torch

    inputs = processor.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, tokenize=True
    )
    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) else inputs.shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Try structured parsing
    triage = parse_triage_response(response)
    if triage:
        triage, _red_flags = escalate_for_safety(triage, symptoms or "")
        display = format_triage_for_display(triage)
    else:
        display = response

    # Update history
    new_history = history + [
        {"role": "user", "content": symptoms or "[Image uploaded]"},
        {"role": "assistant", "content": display},
    ]

    return new_history, new_history


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="MediScan — AI Medical Triage",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .disclaimer { color: #e74c3c; font-size: 14px; padding: 10px;
                      border: 1px solid #e74c3c; border-radius: 5px; }
        """
    ) as demo:

        gr.HTML("""
        <div class="header">
            <h1>🏥 MediScan</h1>
            <h3>AI-Powered Medical Triage Assistant</h3>
            <p>Powered by Google Gemma 4 — Works offline, private, accessible</p>
        </div>
        """)

        gr.HTML("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> MediScan is for informational purposes only
            and is NOT a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare professional.
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Upload Photo of Condition",
                    type="pil",
                    height=300,
                )
                symptoms_input = gr.Textbox(
                    label="📝 Describe Symptoms",
                    placeholder="e.g., Red itchy rash on arm for 3 days, spreading...",
                    lines=3,
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

                gr.Markdown("""
                ### How to Use
                1. **Take a photo** of the skin condition (or upload one)
                2. **Describe symptoms** — when it started, pain level, spreading?
                3. Click **Analyze** for triage guidance
                4. Follow the recommended next steps
                """)

            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="MediScan Assessment",
                    height=500,
                    type="messages",
                )

        # State for conversation history
        state = gr.State([])

        # Event handlers
        analyze_btn.click(
            fn=analyze_condition,
            inputs=[image_input, symptoms_input, state],
            outputs=[chatbot, state],
        )

        clear_btn.click(
            fn=lambda: ([], []),
            outputs=[chatbot, state],
        )

        # Example cases
        gr.Examples(
            examples=[
                [None, "I have a red, circular rash on my arm. It's been there for about a week and is slightly raised. It doesn't itch much but is spreading slowly."],
                [None, "My child has small blisters on their hands and feet. They also have a mild fever. Started 2 days ago."],
                [None, "I burned my hand on a cooking fire yesterday. The skin is red and there are some blisters forming. It hurts a lot."],
                [None, "There are small itchy bumps between my fingers and on my wrists. They get worse at night. Other family members have the same problem."],
            ],
            inputs=[image_input, symptoms_input],
            label="Example Cases (click to try)",
        )

        gr.Markdown("""
        ---
        **MediScan** — Built for the Gemma 4 Good Hackathon | Powered by Google Gemma 4 E4B
        | Healthcare in your pocket — no internet required
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
