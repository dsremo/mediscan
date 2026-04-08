# MediScan: Offline AI-Powered Medical Triage for Underserved Communities

## The Problem

According to the World Health Organization, 3.5 billion people — nearly half the global population — lack access to essential health services. In Sub-Saharan Africa, there is less than one dermatologist per million people, yet 80% of the pediatric population suffers from untreated skin conditions including eczema, scabies, fungal infections, and impetigo. Community health workers (CHWs) in these regions serve as the frontline of healthcare delivery, but they often lack the training to assess skin conditions and must refer patients to distant facilities — a journey many patients cannot afford.

Existing mobile health applications like Ada Health and Babylon require a stable internet connection, making them unusable in the rural communities that need them most. They also transmit sensitive health data to cloud servers, raising privacy concerns in regions where data protection frameworks are still developing.

MediScan was built to close this gap.

## The Solution

MediScan is an AI-powered medical triage assistant that runs entirely on-device using Google Gemma 4. A community health worker can photograph a skin condition, describe symptoms in their own words, and receive structured triage guidance — all without internet access.

The system provides four urgency levels (low, moderate, high, emergency), specific first-aid instructions culturally relevant to tropical and resource-limited settings, and clear guidance on when professional care is needed. Every response includes a medical disclaimer and encourages professional consultation.

## How We Used Gemma 4

MediScan leverages three core capabilities of Gemma 4 that make it uniquely suited for this application:

**Multimodal Understanding.** Gemma 4 E4B natively processes both images and text in a single forward pass. When a CHW photographs a skin lesion and describes "itchy bumps between my fingers, worse at night," the model analyzes both inputs simultaneously. This multimodal capability is critical because many skin conditions look similar in photographs but have distinctive symptom profiles that text descriptions reveal.

**Native Function Calling.** We implemented three structured tools using Gemma 4's native function calling: `analyze_skin_condition` (returns structured JSON with condition name, confidence, urgency, first-aid steps, and follow-up questions), `find_nearest_clinic` (locates health facilities when connectivity is available), and `set_medication_reminder` (creates follow-up reminders for treatment adherence). Function calling ensures outputs are structured and machine-readable, enabling downstream integration with health information systems.

**Edge Deployment.** Gemma 4 E4B's 4.5 billion parameters deliver strong multimodal reasoning while fitting comfortably on consumer devices. Quantized to 4-bit precision using Unsloth, the model requires approximately 3GB of RAM and achieves 12-20 tokens per second on a Snapdragon 8 Gen 3 smartphone. This makes real-time triage feasible on the Android devices already prevalent among CHWs in developing regions.

## Technical Architecture

The system consists of four components:

**1. Perception Layer.** Images are preprocessed and passed to Gemma 4's vision encoder alongside the text prompt. We designed the system prompt to prioritize conditions prevalent in tropical and underserved regions — scabies, fungal infections, impetigo, burns from cooking fires, and tropical ulcers — rather than conditions common in high-income dermatology datasets.

**2. Triage Engine.** Gemma 4 produces structured JSON via function calling, classified into four urgency levels. The urgency classification was designed in consultation with WHO triage guidelines for primary healthcare facilities. LOW indicates self-care at home. MODERATE requires a health worker visit within days. HIGH requires a doctor within 24 hours. EMERGENCY requires immediate transfer.

**3. Knowledge Base.** The system includes an embedded knowledge base of first-aid protocols for 22 skin conditions, sourced from WHO and MSF clinical guidelines. When Gemma 4's function call identifies a condition, the knowledge base provides culturally adapted first-aid steps — for example, recommending coconut oil (widely available and affordable) rather than specialized dermatological creams that may be unavailable.

**4. Conversation Manager.** MediScan maintains conversation history to ask follow-up questions when the initial image and description are insufficient for confident triage. If the model's confidence is below 60%, it generates targeted questions: "How long have you had this condition?" "Does anyone else in the household have similar symptoms?" "Is there fever?" This iterative approach mirrors the clinical history-taking process.

## Fine-Tuning with Unsloth

We fine-tuned Gemma 4 E4B using Unsloth's efficient LoRA implementation on the Skin Disease Dataset — a CC0-licensed collection of 22 skin condition categories. Training used LoRA rank 16 with a learning rate of 2e-4 for 3 epochs on Kaggle's T4x2 GPU. The fine-tuned model shows improved accuracy on conditions common in tropical regions compared to the base model, particularly for scabies, fungal infections, and burns.

We specifically designed training examples in the structured function-calling format, teaching the model to output JSON-structured triage assessments rather than free-form text. This ensures consistent, parseable outputs that can be integrated into health information systems.

## Privacy and Safety

MediScan processes all data on-device. No images, symptoms, or personal health information are transmitted to any server. This privacy-first architecture is essential for healthcare applications in regions where data protection laws may not fully protect patient information, and where cultural sensitivity around photographing medical conditions demands discretion.

Every response includes a mandatory disclaimer that MediScan is not a substitute for professional medical care. The urgency classification system is deliberately conservative — when in doubt, the system recommends seeking professional care rather than self-treatment.

## Impact and Scalability

MediScan addresses a specific, measurable gap: providing preliminary skin condition assessment to communities without dermatological expertise. The target deployment scenario is a CHW in rural Sub-Saharan Africa using an Android smartphone to screen patients during community health visits. By providing immediate triage guidance, MediScan can help CHWs prioritize referrals, reducing the burden on distant health facilities and ensuring that the most urgent cases receive timely attention.

The system is designed for scalability. The Gradio-based interface supports multiple languages (a planned extension), and the modular architecture allows the knowledge base to be expanded with region-specific conditions and treatment protocols. The entire system — model weights, knowledge base, and application — fits on a single smartphone with no ongoing infrastructure costs.

## Live Demo and Code

- **Live Demo:** https://huggingface.co/spaces/Dsremo/mediscan
- **Code Repository:** https://github.com/dsremo/mediscan
- **Video Demo:** [YouTube link]

## Conclusion

MediScan demonstrates that frontier AI capabilities — multimodal understanding, structured reasoning, and efficient edge deployment — can be directed toward solving real healthcare challenges in the communities that need them most. By combining Gemma 4's native capabilities with careful medical system design, we've built a tool that is not just technically impressive but genuinely useful: healthcare in your pocket, no internet required.
