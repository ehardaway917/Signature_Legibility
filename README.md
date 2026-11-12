# Predicting Signature Legibility for Proof-of-Delivery Review

DTSC 5505-401 course project (final report, November 2026).

An illegible signature on a proof-of-delivery (POD) document cannot confirm who received a shipment. This project tests whether a model can flag illegible signatures from the image alone, so a driver could ask for a printed name before leaving.

No labeled legibility dataset exists, so labels come from OCR. Each signature is read with TrOCR and the reading is compared with the signer's known name. A signature counts as **legible** when the letter similarity (1 − normalized Levenshtein distance) is 0.8 or more.

## Main result

| | Test PR-AUC [95% CI] | Test ROC-AUC |
| --- | --- | --- |
| **Selected model: CNN (words)** | **0.575 [0.367, 0.757]** | **0.779** |
| No-skill (majority baseline) | 0.311 | 0.500 |

The test set has 254 signatures from 23 signers that no model saw during training or selection. Intervals come from 2,000 bootstrap resamples drawn by signer group.

- **Stage 1 (trained on signatures):** no model beat a one-feature ink-contrast shortcut in signer-grouped cross-validation.
- **Stage 2 (trained on 35,666 IMGUR5K handwritten words):** the words were labeled with the same OCR rule, with contrast normalized and word length balanced. Both shortcuts fell to chance, and the CNN was the clear choice on the development signatures.

The model was chosen by a fixed rule (highest dev PR-AUC, baselines excluded) and locked in `selected_model.json` on 2026-11-10 at 17:33, before any Stage 2 model was scored on the test set.

## Final model

`cnn_words_hard_s42.pt`, `cnn_words_hard_s43.pt`, `cnn_words_hard_s44.pt`

These are the same CNN trained with three seeds on the 0/1 labels. Reported metrics average the three runs. `retrain_selected_cnn` rebuilds them from the settings in `selected_model.json`.

Architecture: four convolution blocks (16, 32, 64 and 128 filters; 3 × 3 conv, batch norm, ReLU, 2 × 2 max-pool), global average pooling, dropout 0.3 and one output. Input is a 128 × 128 grayscale image, contrast-stretched and inverted (ink high, paper zero).

The fine-tuned and soft-target CNN variants in the report's Stage 2 table are not included here; they are not the selected model.

## How to run

Run the notebooks top to bottom, in this order:

| # | Notebook | What it does | Main outputs |
| --- | --- | --- | --- |
| 1 | `label_signatures` | Reads each signature with TrOCR and computes similarity to the known name | `signatures_labels.csv` |
| 2 | `Split_Data` | Groups signers, then makes the dev/test split and grouped 5-fold CV (seeds 42, 43, 44) | `splits.csv` |
| 3 | `model_comparison` | Stage 1: baselines, 4 classical models and the CNN, cross-validated on dev | `model_comparison_summary.csv` |
| 4 | `holdout_eval` | Stage 1: refits on all of dev and scores the test set once | `holdout_results.csv`, `holdout_predictions.csv` |
| 5 | `word_legibility_pipeline` | Downloads and crops IMGUR5K words, labels them with TrOCR and balances word length | `words_labels.csv` |
| 6 | `train_word_models_soft` | Stage 2: trains on words, selects on dev signatures, scores the test set | `selected_model.json`, `word_models_selection.csv`, `word_models_holdout.csv`, `soft_vs_hard_dev.csv` |
| 7 | `retrain_selected_cnn` | Retrains the selected model and saves its weights | `cnn_words_hard_s42/43/44.pt` |

Seeds are fixed: 42 for splits and sampling, and 42/43/44 for CV folds and CNN runs.

## Data

The image data are not included in this repository. Download them from the original sources:

- **Signatures:** [sl4shuur/handwriting-signatures-dataset](https://huggingface.co/datasets/sl4shuur/handwriting-signatures-dataset) on Hugging Face (1,289 images, CC BY-NC 4.0). `label_signatures` builds `signature_images_256.npy` from it.
- **Handwritten words:** IMGUR5K (Krishnan et al., 2023). `word_legibility_pipeline` downloads the photos with the dataset's own script and verifies their checksums.

Included: OCR labels, the signer-grouped splits, the result tables and the selection record. Signer names come from the public dataset captions. Results are reported only in aggregate.

## Environment

Windows, NVIDIA GeForce RTX 5060 Laptop GPU, Python 3.14, PyTorch 2.15 (CUDA 13.4), Transformers 5.22, scikit-learn, scikit-image 0.26, pandas 3.1 and NumPy 2.5. Install with:

```
pip install -r requirements.txt
```

TrOCR (`microsoft/trocr-base-handwritten`) was run in float16 with greedy decoding.

## Repository layout

All files sit in one folder:

- **Notebooks:** the 7 listed above
- **Labels and splits:** `signatures_labels.csv` (final labels), `signatures_labeled.csv` (earlier labels, used for the Stage 1 test results), `splits.csv`, `words_labels.csv`, `image_stats_v2.csv`
- **Selection record:** `selected_model.json`
- **Results:** `model_comparison_summary.csv`, `holdout_results.csv`, `holdout_predictions.csv`, `word_models_selection.csv`, `word_models_holdout.csv`, `soft_vs_hard_dev.csv`
- **Final model:** `cnn_words_hard_s42.pt`, `cnn_words_hard_s43.pt`, `cnn_words_hard_s44.pt`
- `requirements.txt`, `README.md`

## Limitations

- The labels measure whether TrOCR can read the name, not whether a person can.
- The test set has only 23 signers, so the intervals are wide.
- All signatures come from one dataset. The model has not yet been tested on real POD signatures (Tobacco800) or against the human-labeled set.

See the final report for details.
