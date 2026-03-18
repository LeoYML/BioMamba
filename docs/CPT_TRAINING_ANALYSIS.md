## Continual Pre-Training

We perform continual pre-training (CPT) on five Mamba2 model variants ranging from 130M to 2.7B parameters. All models are initialized from their respective pre-trained checkpoints and further trained on a mixture of PubMed abstracts and Wikipedia articles using the single-document packing strategy. We use the GPT-NeoX-20B tokenizer across all model sizes for consistency.

### Training Setup

| Model | Parameters | Max LR | Warmup | Scheduler | Total Steps | Effective Batch Size |
|-------|-----------|--------|--------|-----------|-------------|---------------------|
| Mamba2-130m | 130M | 1.5e-6 | 500 | Cosine | 6,000 | 32 |
| Mamba2-370m | 370M | 4.5e-7 | 500 | Cosine | 6,000 | 32 |
| Mamba2-780m | 780M | 3.0e-7 | 500 | Cosine | 6,000 | 32 |
| Mamba2-1.3b | 1.3B | 2.5e-7 | 500 | Cosine | 6,000 | 32 |
| Mamba2-2.7b | 2.7B | 2.0e-7 | 500 | Cosine | 8,000 | 32 |

We adopt conservatively small learning rates with cosine decay to prevent catastrophic forgetting of general language capabilities while adapting the models to biomedical domain knowledge.

### Training Dynamics

Figure 1 and Figure 2 show the validation loss and validation perplexity curves during CPT across all five model sizes.

![Validation Loss](plot_cpt_eval_loss.png)
*Figure 1: Validation loss during continual pre-training. Larger models consistently achieve lower validation loss, and all models show steady convergence throughout training.*

![Validation Perplexity](plot_cpt_eval_ppl.png)
*Figure 2: Validation perplexity during continual pre-training. The 2.7B model achieves the lowest perplexity of 5.20, while even the smallest 130M model converges to a perplexity of 8.35.*

### Results

| Model | Best Val Loss | Best Val PPL | Improvement over Init |
|-------|--------------|-------------|----------------------|
| Mamba2-130m | 2.123 | 8.35 | - |
| Mamba2-370m | 1.897 | 6.66 | - |
| Mamba2-780m | 1.800 | 6.05 | - |
| Mamba2-1.3b | 1.720 | 5.59 | - |
| Mamba2-2.7b | 1.649 | 5.20 | - |

Several observations emerge from the training curves:

1. **Consistent scaling behavior.** Larger models achieve uniformly lower validation loss and perplexity across all training steps. The ordering 130m > 370m > 780m > 1.3b > 2.7b is maintained throughout the entire training process, consistent with neural scaling laws.

2. **Smooth convergence.** All models exhibit monotonically decreasing validation loss curves without signs of overfitting or training instability, suggesting that the chosen learning rates and data mixture are well-calibrated for continual pre-training.

3. **Diminishing marginal returns.** The perplexity gap between adjacent model sizes narrows as models grow larger: the gap between 130m and 370m is 1.69 PPL, while the gap between 1.3b and 2.7b is only 0.39 PPL. This is consistent with the log-linear scaling behavior observed in large language models.

4. **Efficient adaptation.** Most of the perplexity improvement occurs within the first 2,000 training steps, after which the curves plateau. This indicates that the domain adaptation from general-purpose to biomedical text is achieved efficiently with relatively few gradient updates, a desirable property that limits the computational cost of CPT.
