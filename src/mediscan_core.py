"""
MediScan — Offline Medical Triage Assistant powered by Gemma 4

Core inference pipeline:
1. Accept image (skin condition photo) + text (symptom description)
2. Gemma 4 multimodal analysis
3. Structured triage output via function calling
4. Follow-up questions if needed
5. First-aid guidance + urgency classification

Runs on: Kaggle GPU (T4x2), HuggingFace Spaces, or any device with Ollama/llama.cpp
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class UrgencyLevel(Enum):
    LOW = "low"          # Self-care at home
    MODERATE = "moderate" # See a health worker within days
    HIGH = "high"        # See a doctor within 24 hours
    EMERGENCY = "emergency" # Seek immediate medical care


@dataclass
class TriageResult:
    """Structured output from MediScan analysis."""
    condition_name: str
    confidence: float  # 0-1
    urgency: UrgencyLevel
    description: str
    first_aid: List[str]
    when_to_seek_care: str
    follow_up_questions: List[str]
    disclaimer: str = ("This is for informational purposes only and is NOT a medical "
                       "diagnosis. Always consult a qualified healthcare professional.")


# ===== Function Calling Schema for Gemma 4 =====

MEDISCAN_TOOLS = [
    {
        "name": "analyze_skin_condition",
        "description": "Analyze a photo of a skin condition and provide triage guidance",
        "parameters": {
            "type": "object",
            "properties": {
                "condition_name": {
                    "type": "string",
                    "description": "Most likely skin condition name"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "moderate", "high", "emergency"],
                    "description": "Urgency level for seeking care"
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the condition"
                },
                "first_aid_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Immediate first-aid steps"
                },
                "when_to_seek_care": {
                    "type": "string",
                    "description": "When should the patient see a doctor"
                },
                "follow_up_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Questions to ask for better assessment"
                }
            },
            "required": ["condition_name", "confidence", "urgency", "description",
                         "first_aid_steps", "when_to_seek_care"]
        }
    },
    {
        "name": "find_nearest_clinic",
        "description": "Find the nearest health facility based on location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "condition_type": {"type": "string"}
            }
        }
    },
    {
        "name": "set_medication_reminder",
        "description": "Set a reminder for medication or follow-up",
        "parameters": {
            "type": "object",
            "properties": {
                "medication": {"type": "string"},
                "frequency": {"type": "string"},
                "duration_days": {"type": "integer"},
                "notes": {"type": "string"}
            }
        }
    }
]


SYSTEM_PROMPT = """You are MediScan, an AI-powered medical triage assistant designed for
community health workers in underserved areas. You help assess skin conditions from photos
and provide preliminary triage guidance.

CRITICAL RULES:
1. You are NOT a doctor. Always include a disclaimer.
2. When in doubt, recommend seeking professional care.
3. Be culturally sensitive and use simple, clear language.
4. Consider conditions common in tropical/developing regions.
5. Ask follow-up questions when the image alone is insufficient.
6. Classify urgency accurately — lives may depend on it.

URGENCY LEVELS:
- LOW: Minor condition, self-care at home (e.g., mild dry skin, small insect bite)
- MODERATE: Should see a health worker within a few days (e.g., suspected fungal infection)
- HIGH: Should see a doctor within 24 hours (e.g., infected wound, spreading rash)
- EMERGENCY: Seek immediate medical care (e.g., severe burns, signs of sepsis)

When analyzing an image, use the analyze_skin_condition function to provide structured output.
If the user describes symptoms without an image, ask for a photo if possible, but still provide
preliminary guidance based on the description.

Common conditions in underserved tropical regions:
- Scabies, eczema, fungal infections (tinea), impetigo
- Wounds and burns (often from cooking fires)
- Insect bites and stings
- Allergic reactions
- Tropical ulcers, leishmaniasis
"""


def build_chat_messages(
    user_text: str,
    image_path: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None
) -> List[Dict]:
    """Build the chat message sequence for Gemma 4.

    Supports multimodal input (image + text) and conversation history.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Build current user message
    if image_path:
        # Multimodal: image + text
        content = [
            {"type": "image", "source": image_path},
            {"type": "text", "text": user_text or "Please analyze this skin condition."}
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


def parse_triage_response(response_text: str) -> Optional[TriageResult]:
    """Parse Gemma 4's function call response into a TriageResult."""
    try:
        # Try to extract JSON from function call
        if "analyze_skin_condition" in response_text:
            # Extract JSON between braces
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            data = json.loads(response_text[start:end])

            return TriageResult(
                condition_name=data.get("condition_name", "Unknown"),
                confidence=float(data.get("confidence", 0.5)),
                urgency=UrgencyLevel(data.get("urgency", "moderate")),
                description=data.get("description", ""),
                first_aid=data.get("first_aid_steps", []),
                when_to_seek_care=data.get("when_to_seek_care", "Consult a healthcare professional"),
                follow_up_questions=data.get("follow_up_questions", []),
            )
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return None


def format_triage_for_display(result: TriageResult) -> str:
    """Format a TriageResult for human-readable display."""
    urgency_emoji = {
        UrgencyLevel.LOW: "🟢",
        UrgencyLevel.MODERATE: "🟡",
        UrgencyLevel.HIGH: "🟠",
        UrgencyLevel.EMERGENCY: "🔴",
    }

    lines = [
        f"## {urgency_emoji[result.urgency]} MediScan Assessment",
        f"**Condition:** {result.condition_name} (confidence: {result.confidence:.0%})",
        f"**Urgency:** {result.urgency.value.upper()}",
        f"",
        f"### Description",
        result.description,
        f"",
        f"### First Aid Steps",
    ]
    for i, step in enumerate(result.first_aid, 1):
        lines.append(f"{i}. {step}")

    lines.extend([
        f"",
        f"### When to Seek Care",
        result.when_to_seek_care,
    ])

    if result.follow_up_questions:
        lines.extend([
            f"",
            f"### Follow-up Questions",
        ])
        for q in result.follow_up_questions:
            lines.append(f"- {q}")

    lines.extend([
        f"",
        f"---",
        f"*{result.disclaimer}*",
    ])

    return "\n".join(lines)


# ===== Kaggle Notebook Integration =====

def load_gemma4_kaggle():
    """Load Gemma 4 model on Kaggle notebook with GPU.

    Uses transformers + bitsandbytes for 4-bit quantization.
    Returns (model, processor) tuple.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        model_id = "google/gemma-4-E4B"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        return model, processor
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. "
            "Run: pip install transformers bitsandbytes accelerate"
        )


def run_mediscan_inference(
    model,
    processor,
    user_text: str,
    image=None,
    conversation_history=None,
    max_new_tokens=1024,
):
    """Run MediScan inference with Gemma 4.

    Args:
        model: Loaded Gemma 4 model
        processor: Gemma 4 processor/tokenizer
        user_text: User's text input
        image: PIL Image or None
        conversation_history: List of previous messages
        max_new_tokens: Max tokens to generate

    Returns:
        (response_text, triage_result) tuple
    """
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    if image is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text or "Please analyze this skin condition."}
            ]
        })
    else:
        messages.append({"role": "user", "content": user_text})

    # Add tool definitions
    tools_text = f"\n\nAvailable tools:\n{json.dumps(MEDISCAN_TOOLS, indent=2)}"
    messages[0]["content"] += tools_text

    # Process with Gemma 4
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    # Decode only the new tokens
    if isinstance(inputs, dict):
        input_len = inputs["input_ids"].shape[-1]
    else:
        input_len = inputs.shape[-1]

    response_text = processor.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )

    # Try to parse structured triage output
    triage = parse_triage_response(response_text)

    return response_text, triage
