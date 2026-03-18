"""
Prompt templates for MIMIC clinical evaluation tasks.

Note: The CPT model is a pure language model (not instruction-tuned).
- PPL and completion tasks work best (no instruction needed).
- Classification tasks (mortality) use logprob comparison, not generation.
- Instruction-based tasks (discharge gen) work better after clinical SFT.
"""

# ---------------------------------------------------------------------------
# Completion: pure continuation, no instruction wrapper
# ---------------------------------------------------------------------------
COMPLETION_TEMPLATE = "{prefix}"

# ---------------------------------------------------------------------------
# Mortality prediction (yes = survived, no = died)
# Used with logprob comparison P(yes) vs P(no)
# ---------------------------------------------------------------------------
MORTALITY_TEMPLATE = (
    "Based on the following clinical notes from the first 48 hours of admission, "
    "predict whether the patient will survive to hospital discharge. "
    "Answer with yes or no.\n\n"
    "Clinical Notes:\n{context}\n\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Discharge summary generation
# ---------------------------------------------------------------------------
DISCHARGE_TEMPLATE = (
    "Based on the following clinical information, write the discharge summary.\n\n"
    "Clinical Notes:\n{context}\n\n"
    "Discharge Summary:"
)

# ---------------------------------------------------------------------------
# Few-shot mortality prompt (for base / CPT models without SFT)
# ---------------------------------------------------------------------------
MORTALITY_FEWSHOT_TEMPLATE = (
    "Predict hospital survival from clinical notes. Answer: yes (survived) or no (died).\n\n"
    "Example 1:\nClinical Notes: Patient admitted for elective hip replacement. "
    "Post-op course uneventful.\nAnswer: yes\n\n"
    "Example 2:\nClinical Notes: Patient admitted with multi-organ failure, "
    "refractory hypotension despite vasopressors.\nAnswer: no\n\n"
    "Clinical Notes:\n{context}\n\n"
    "Answer:"
)


def format_completion_prompt(prefix_text: str) -> str:
    return COMPLETION_TEMPLATE.format(prefix=prefix_text)


def format_mortality_prompt(
    context: str, max_ctx_chars: int = 4000, use_fewshot: bool = False
) -> str:
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    template = MORTALITY_FEWSHOT_TEMPLATE if use_fewshot else MORTALITY_TEMPLATE
    return template.format(context=trunc)


def format_discharge_prompt(context: str, max_ctx_chars: int = 4000) -> str:
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    return DISCHARGE_TEMPLATE.format(context=trunc)
