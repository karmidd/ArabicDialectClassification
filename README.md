# Arabic Dialect Identification via Fine-Tuning Pre-Trained ASR Models

Fine-grained classification of 17 Arabic dialects by repurposing the audio encoder of [Qwen3-ASR](https://arxiv.org/abs/2601.21337) with Low-Rank Adaptation (LoRA). Achieves **93.78% test accuracy** on the ADI-17 dataset using only the 0.6B parameter model — outperforming a from-scratch MLP baseline (14.34%) by a factor of 6.5×.

**Course:** COMP 432 — Machine Learning (Concordia University)  
**Instructor:** Dr. Mirco Ravanelli | **Supervisor:** Maab Elrashid  
**Author:** Karam Midani

---

## Key Findings

- **Foundation models dominate:** Frozen Qwen3-ASR encoders reach ~69% accuracy with zero fine-tuning, while an MLP trained from scratch on the same Mel-spectrograms plateaus at 14%.
- **LoRA capacity bottleneck:** Low-rank adapters (r=8) collapse catastrophically at high data volumes (dropping to ~40% accuracy), while high-rank adapters (r=32) scale smoothly to 93%+.
- **Smaller model wins on efficiency:** The 0.6B model matches the 1.7B model's peak accuracy (93.78% vs 93.70%) at a fraction of the compute, making it the optimal choice for this task.
- **Errors are linguistically meaningful:** The confusion matrix shows the highest misclassification rate between Egyptian and Sudanese dialects — geographically contiguous regions with shared phonetic traits.

## Architecture

```
Raw Audio (16kHz) → Qwen3 Processor → 128-dim Mel-Spectrograms
    → AuT Encoder (Transformer + Conv Front-End)
        → LoRA Adapters (q_proj, v_proj, r=32)
    → Mean Pooling → MLP Head (hidden → 256 → 17) → Dialect Prediction
```

The Qwen3-ASR model is a Large Audio-Language Model built on Qwen3-Omni. We discard its text decoder and projector, extracting only the AuT audio encoder — pre-trained on ~40M hours of speech — and attach a lightweight classification head.

## Results

| Strategy | 50/d | 100/d | 200/d | 500/d | 1000/d |
|---|---|---|---|---|---|
| MLP Baseline | 7.42 | 7.54 | 7.42 | 11.16 | 14.34 |
| 0.6B Frozen | 23.13 | 28.23 | 46.39 | 59.58 | 56.98 |
| 0.6B LoRA r=8 | 19.14 | 38.89 | 74.38 | 52.11 | 47.51 |
| **0.6B LoRA r=32** | 63.06 | 57.21 | 88.56 | **93.78** | 92.33 |
| 1.7B Frozen | 26.87 | 43.70 | 47.94 | 61.53 | 69.14 |
| 1.7B LoRA r=8 | 15.46 | 19.51 | 79.24 | 40.59 | 53.43 |
| **1.7B LoRA r=32** | 28.87 | 61.03 | 71.20 | 89.59 | **93.70** |

*Test accuracy (%) across balanced subsets of the ADI-17 dataset.*

## Dataset

[ADI-17](https://swshon.github.io/pdf/shon_2020_adi17.pdf) — Arabic Dialect Identification dataset covering 17 dialects: ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SUD, SYR, UAE, YEM.

To handle severe class imbalance, we created strictly balanced subsets (50–1000 samples per dialect) with a fixed 80/10/10 train/val/test split. For each data scale, the encoder was re-initialized from the original pretrained checkpoint.

## Repository Structure

```
├── Final_Report.ipynb   # Full notebook (report + runnable code)
├── visuals/
│   ├── architecture_diagram.png
│   ├── tsne_frozen.jpg           # t-SNE before fine-tuning
│   ├── tsne_embeddings.png       # t-SNE after fine-tuning (r=32)
│   └── confusion_matrix.png
└── README.md
```

### Checkpoints

Trained model checkpoints are available for download in the [Releases](https://github.com/karmidd/ArabicDialectClassification/releases) tab. To reproduce evaluation results without retraining, download the `.pt` file and load it:

```python
model.load_state_dict(torch.load("best_lora_0_6B_500pd_r32_lr5e5.pt"))
model.eval()
```

## Environment

Training was performed locally on:
- **GPU:** AMD Radeon RX 7800 XT (16GB VRAM)
- **OS:** WSL2 / Ubuntu 24.04
- **Framework:** PyTorch 2.9 + ROCm 7.2
- **Key libraries:** `transformers`, `peft`, `qwen-asr`, `scikit-learn`

ROCm stability flags used:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
```

## References

1. S. Shon et al., "ADI17: A fine-grained Arabic dialect identification dataset," ICASSP, 2020.
2. X. Shi et al., "Qwen3-ASR technical report," arXiv:2601.21337, 2026.
3. E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685, 2021.
4. H. A. Alsayadi et al., "Dialectal Arabic speech recognition using CNN-LSTM," ICCAE, 2022.
5. A. Ali et al., "ADI-20: Arabic dialect identification dataset and models," arXiv:2511.10070, 2025.
6. A. Radford et al., "Robust speech recognition via large-scale weak supervision," ICML, 2023.
