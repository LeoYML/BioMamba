# Tokenizer Information for BioMamba2 Project

## Overview

This document explains which tokenizers are used across different models in the BioMamba2 project.

## Tokenizer Details

### GPT-NeoX Tokenizer
- **Source**: `state-spaces/mamba-2.8b-hf`
- **Type**: `GPTNeoXTokenizer`
- **Vocab Size**: ~50,277 tokens
- **Special Tokens**:
  - `<|endoftext|>` - Used for BOS/EOS/PAD/UNK

## Tokenizer Usage by Model

### 1. Bio Pre-trained Mamba2 Models
**Path**: `./checkpoints/biomamba2_mamba2-130m_20260203_083540/`

- ✅ **Includes tokenizer**: Yes (saved during training)
- **Source**: `state-spaces/mamba-2.8b-hf`
- **Used in**: `finetune_pubmed_medline.py` (line 253)
- **Training**: Pre-trained on PubMed-MEDLINE dataset

```python
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
```

### 2. SFT Fine-tuned Mamba2 Models
**Path**: `./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/`

- ✅ **Includes tokenizer**: Yes (inherited from pre-trained model)
- **Source**: Same as pre-trained model
- **Training**: Fine-tuned on PubMedQA for question answering

### 3. Original HuggingFace Mamba Models
**Paths**: 
- `state-spaces/mamba2-130m`
- `state-spaces/mamba2-370m`
- `state-spaces/mamba2-780m`

- ❌ **Includes tokenizer**: No
- **Fallback**: Automatically uses `state-spaces/mamba-2.8b-hf` tokenizer
- **Reason**: Original models don't ship with tokenizers

```python
# Automatic fallback in evaluate_bioasq.py
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except:
    if 'state-spaces' in model_path:
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
```

### 4. BioGPT Model
**Path**: `microsoft/biogpt`

- ✅ **Includes tokenizer**: Yes
- **Type**: `BioGptTokenizer`
- **Vocab Size**: ~42,384 tokens (biomedical-specific)
- **Note**: Different from Mamba models

## Evaluation Scripts

### Tokenizer Consistency

All evaluation scripts ensure tokenizer consistency:

| Script | Model | Tokenizer Used |
|--------|-------|----------------|
| `run_evaluate_bioasq.sh` | SFT Fine-tuned | Included in checkpoint |
| `run_evaluate_mamba_nosft.sh` | Bio Pre-trained | Included in checkpoint |
| `run_evaluate_mamba_original.sh` | Original HF Mamba | `state-spaces/mamba-2.8b-hf` (fallback) |
| `run_evaluate_biogpt.sh` | BioGPT | `microsoft/biogpt` |

### Why Use mamba-2.8b-hf Tokenizer?

1. **Consistency**: Same tokenizer across all Mamba models
2. **Compatibility**: Works with all state-spaces Mamba models
3. **Pre-training**: Used in our bio pre-training pipeline
4. **Vocabulary**: Rich vocabulary suitable for biomedical text

## Important Notes

### For Original HuggingFace Mamba Models

When evaluating `state-spaces/mamba2-*` models:
- The script will automatically download and use `state-spaces/mamba-2.8b-hf` tokenizer
- First run may take a few minutes to download
- This ensures fair comparison with bio pre-trained models

### For Local Checkpoints

When evaluating local checkpoints:
- Tokenizer is already saved in the checkpoint directory
- No additional downloads needed
- Tokenizer config preserved from training

## Verification

To verify which tokenizer a model is using:

```bash
# Check tokenizer config
cat checkpoints/biomamba2_mamba2-130m_20260203_083540/final_model/tokenizer_config.json

# Look for "tokenizer_class": "GPTNeoXTokenizer"
```

## Troubleshooting

### Issue: "Can't load tokenizer"
**Solution**: Model doesn't include a tokenizer. The evaluation script will automatically use `state-spaces/mamba-2.8b-hf` as fallback.

### Issue: "Token IDs out of range"
**Solution**: Ensure you're using the correct tokenizer. Bio models should use GPT-NeoX tokenizer (~50K vocab), not a smaller one.

### Issue: Different results between models
**Possible cause**: Using different tokenizers. Verify tokenizer consistency across models being compared.

## Summary

- **Bio Pre-trained & SFT Models**: Use GPT-NeoX tokenizer from `state-spaces/mamba-2.8b-hf`
- **Original HF Mamba Models**: Automatically use same GPT-NeoX tokenizer for consistency
- **BioGPT**: Uses its own biomedical tokenizer
- **All models**: Tokenizer selection is handled automatically by evaluation scripts
