# Signal Separation and Classification in Noisy Environments with Machine Learning

> A convolutional neural network pipeline that detects and counts Capuchinbird calls in continuous forest audio recordings by classifying mel spectrogram segments.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Fall%202024-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Machine Learning Pattern Recognition                                                                                                                                                                                                                                                                                                                      |
| Course Number          | EEL 5825                                                                                                                                                                                                                                                                                                                                                  |
| Semester               | Fall 2024                                                                                                                                                                                                                                                                                                                                                 |
| Assignment Title       | Final Course Project                                                                                                                                                                                                                                                                                                                                      |
| Assignment Description | Design and implement a complete machine learning system that demonstrates mastery of core ML principles, including model architecture, training methodology, generalization, and evaluation. The project should reflect the theoretical foundations of why machine learning works, applying learned concepts to a real-world pattern recognition problem. |

---

## Project Description

This project applies convolutional neural network-based binary classification to the problem of detecting Capuchinbird vocalizations within continuous, noisy rainforest recordings. A CNN is trained on mel spectrogram representations of labeled audio clips and then deployed in a sliding-window inference loop to count bird calls across full-length recordings. The system includes data augmentation (pitch shift, time stretch, noise injection), a three-way train/validation/test split, adaptive learning rate scheduling, and automated comparison of predictions against ground-truth call counts.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

When run correctly, the pipeline produces the following outputs in the `Results/` directory:

```
Results/
├── best_model.pth              # Model weights from the epoch with lowest validation loss
├── log.txt                     # Full console output mirrored to disk
├── Results.csv                 # Per-call detail: filename, call number, start/end time, duration, confidence
├── ResultsSummary.csv          # Per-file total call counts (predicted)
├── 1.TrainingHistory.png       # Loss and accuracy curves across all epochs
├── 2.FinalMetrics.png          # Bar chart of final loss, accuracy, precision, recall, F1
└── 3.ModelResultsAccuracy.png  # Scatter plot of predicted vs. ground-truth call counts
```

**Sample terminal output (training phase):**

```
Using device: cuda

Using Augmented Dataset
        Training set size: 1204
        Validation set size: 213
        Test set size: 213

Epoch 1/50 ----------------------------------------------------------------------------------------------------
Current Learning Rate:  0.001000
Ein (Training Error):   0.312451
Eval (Validation Error):0.289034
Eout (Test Error):      0.301122
Training Accuracy:      0.873456
Validation Accuracy:    0.882160
Test Accuracy:          0.877340
```

**Sample terminal output (analysis phase):**

```
Found 24 audio files to analyze
Analyzing recordings: 100%|████████████| 24/24
Results:
MAE:  2.14
RMSE: 3.07
R²:   0.871
```

**Interpreting the outputs:**

- **Ein / Eval / Eout** correspond to training, validation, and test error respectively. Healthy runs show all three converging; a large gap between Ein and Eout indicates overfitting.
- **R² (coefficient of determination)** in the analysis phase measures how well predicted call counts track ground truth across files. Values above 0.85 indicate strong agreement.
- **`3.ModelResultsAccuracy.png`** plots each recording as a point: points near the red dashed line are accurate predictions. Systematic bias (all points above or below the line) suggests the confidence threshold in `analysis.py` (`confidence_threshold = 0.8`) may need adjustment.
- If analysis fails to find `Results/best_model.pth`, run training first (mode 2) before running analysis (mode 3).

---

## Key Concepts

`Convolutional Neural Network` `Mel Spectrogram` `Binary Classification` `Data Augmentation` `Sliding Window Inference` `Transfer Learning` `Batch Normalization` `Dropout Regularization` `Learning Rate Scheduling` `Signal Processing`

---

## Languages & Tools

- **Language:** Python 3.x
- **Framework:** PyTorch, librosa
- **Build System:** pip / requirements.txt

---

## File Structure

```
project-root/
├── main.py                          # Entry point; CLI interface for all pipeline modes
├── requirements.txt                 # Third-party dependencies
├── Data/                            # Raw audio dataset
│   ├── Parsed_Capuchinbird_Clips/   # Labeled positive samples (.wav)
│   ├── Parsed_Not_Capuchinbird_Clips/ # Labeled negative samples (.wav)
│   └── Augmented/                   # Auto-generated augmented dataset (created at runtime)
├── Figures/                         # Waveform and spectrogram demo figures (mode 4)
├── Results/                         # All training outputs, logs, and analysis CSVs
│   └── GroundTruth.csv              # Ground-truth call counts for analysis evaluation
└── src/
    ├── model.py                     # AudioCNN architecture definition
    ├── data_processing.py           # AudioDataset (PyTorch Dataset) and augmentation logic
    ├── training.py                  # Training loop, evaluation, metric tracking, plot generation
    ├── analysis.py                  # Sliding-window inference and call grouping on full recordings
    ├── utils.py                     # Demo figure generation (waveforms and spectrograms)
    └── logging_module.py            # Dual-output logger (console + Results/log.txt)
```

---

## Installation & Usage

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU recommended (CPU fallback supported)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-MLPatternRecognition-SignalClassification.git
cd UCF-MLPatternRecognition-SignalClassification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (train + analyze)
python main.py
```

### Controls

| Argument          | Values                       | Description                                                                |
| ----------------- | ---------------------------- | -------------------------------------------------------------------------- |
| `--mode`          | `1` (default), `2`, `3`, `4` | 1: full pipeline, 2: train only, 3: analyze only, 4: generate demo figures |
| `--epochs`        | integer (default: `50`)      | Number of training epochs                                                  |
| `--learning_rate` | float (default: `0.001`)     | Adam optimizer learning rate                                               |
| `--augment`       | `T` (default), `F`           | Whether to use the augmented dataset                                       |

**Example commands:**

```bash
# Training only, 100 epochs, custom learning rate
python main.py --mode 2 --epochs 100 --learning_rate 0.005

# Analysis only (requires a trained model in Results/)
python main.py --mode 3

# Generate demo figures
python main.py --mode 4
```

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
