# Vehicle Type Classification 🚗🚌🏍️🚚

A transfer-learning image classifier that identifies vehicle types (Car, Bus, Truck, Motorcycle, etc.) using a fine-tuned **ResNet18** in PyTorch. Includes full evaluation metrics — accuracy, precision, recall, F1-score (macro & weighted), confusion matrix, and per-class classification report.

## 🧠 Overview

- **Architecture:** ResNet18 (pretrained on ImageNet), fine-tuned end-to-end
- **Framework:** PyTorch + torchvision
- **Training:** 10 epochs, Adam optimizer, StepLR scheduler
- **Augmentation:** Random horizontal flip, rotation, color jitter (train only)
- **Metrics:** Accuracy, Precision, Recall, F1 (macro + weighted), Confusion Matrix, Classification Report
- **Best model selection:** Checkpoint saved on best validation macro-F1 (not just last epoch)

## 📁 Repository Structure

```
vehicle-type-classification/
├── vehicle_classification.py          # Local/offline script (edit DATA_DIR to your dataset path)
├── vehicle_classification_colab.py    # Google Colab version (Kaggle API download + Drive save)
├── vehicle_classification_colab.ipynb # Same pipeline as a runnable Colab notebook
└── README.md
```

## 📊 Dataset

Any vehicle image dataset organized in `ImageFolder` format works, e.g.:

```
dataset/
├── train/
│   ├── Bus/
│   ├── Car/
│   ├── Motorcycle/
│   └── Truck/
└── val/            # optional — auto 80/20 split from train/ if missing
    ├── Bus/
    ├── Car/
    ├── Motorcycle/
    └── Truck/
```

Suggested source: search "vehicle type" / "vehicle classification" on [Kaggle Datasets](https://www.kaggle.com/datasets) and pick one with clean class-labeled folders.

## 🚀 Usage

### Option A — Local machine
```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm
```
Edit `DATA_DIR` in `vehicle_classification.py` to point to your dataset, then:
```bash
python vehicle_classification.py
```

### Option B — Google Colab (recommended, free GPU)
1. Open `vehicle_classification_colab.ipynb` in Colab (or paste `vehicle_classification_colab.py` into a cell)
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells top to bottom:
   - Upload your `kaggle.json` (Kaggle → Account → Create New API Token) when prompted
   - Set `KAGGLE_DATASET_SLUG` to your chosen dataset
   - Adjust `DATA_DIR` after checking the printed folder structure
4. Trained model + metrics + plots are saved to `/content/outputs` and optionally copied to Google Drive

## 📈 Outputs

After training, you'll get:

| File | Description |
|---|---|
| `best_vehicle_model.pth` | Best model weights (by validation macro-F1) |
| `training_curves.png` | Train/val loss and accuracy/F1 per epoch |
| `confusion_matrix.png` | Confusion matrix heatmap across all classes |
| `metrics_report.txt` | Full accuracy/precision/recall/F1 summary + per-class report |

## 🔧 Configuration

Key parameters (top of script):

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 32 | Batch size |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `IMG_SIZE` | 224 | Input image resolution |
| `FREEZE_BACKBONE` | False | Set `True` to only train the final layer instead of fine-tuning the whole network |

## 🛠️ Tech Stack

- Python 3.11
- PyTorch / torchvision
- scikit-learn (metrics)
- matplotlib / seaborn (visualization)

## 📌 Notes

- Class imbalance is common in vehicle datasets (e.g. far more car images than motorcycles) — macro-F1 is reported alongside accuracy for a fairer picture of per-class performance.
- To improve results further: try a deeper backbone (ResNet34/50), longer training with early stopping, or class-weighted loss if imbalance is severe.

## 📄 License

MIT
