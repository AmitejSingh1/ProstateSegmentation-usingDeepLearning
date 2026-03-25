# 🔬 Prostate Segmentation using Deep Learning

AI-powered prostate boundary segmentation from micro-ultrasound images using a **VGG16-UNet** architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

## Architecture

The model uses a **VGG16 encoder** (pretrained on ImageNet) paired with a **U-Net decoder** featuring skip connections for precise boundary localization:

```
Input (512×512×3)
    │
    ▼
┌─────────────────────────────────────────┐
│           VGG16 Encoder                 │
│  block1_conv2 ──────────────────┐       │
│  block2_conv2 ────────────┐     │       │
│  block3_conv3 ──────┐     │     │       │
│  block4_conv3 ──┐   │     │     │       │
│  block5_conv3   │   │     │     │  Skip │
└────────┬────────┘   │     │     │  Conn │
         │            │     │     │       │
    ┌────▼────┐  ┌────▼───┐ │     │       │
    │ Dec 512 │  │Dec 256 │ │     │       │
    └────┬────┘  └────┬───┘ │     │       │
         │            │  ┌──▼──┐  │       │
         └────────────┘  │D 128│  │       │
                         └──┬──┘  │       │
                            │  ┌──▼──┐    │
                            └──│D 64 │    │
                               └──┬──┘    │
                                  │       │
                            Conv2D(1,1)   │
                             Sigmoid      │
                                  │
                                  ▼
                        Binary Mask (512×512×1)
```

- **Loss Function**: Dice Loss
- **Metrics**: Dice Coefficient, IoU, Accuracy, F1, Precision, Recall

## Project Structure

```
prostate/
├── app.py              # 🌐 Streamlit web application
├── model.py             # 🧠 VGG16-UNet model definition
├── metrics.py           # 📊 Dice loss, Dice coefficient, IoU
├── config.py            # ⚙️  Central configuration
├── train.py             # 🏋️ Training script (CLI)
├── evaluate.py          # 📈 Evaluation with metrics & visuals
├── predict.py           # 🔮 DICOM inference script
├── preprocess.py        # 🔄 NIfTI → PNG slice converter
├── requirements.txt     # 📦 Python dependencies
├── Dockerfile           # 🐳 Container deployment
├── .gitignore
└── files/               # Model weights & logs (gitignored)
    └── model.keras
```

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/AmitejSingh1/ProstateSegmentation-usingDeepLearning.git
cd ProstateSegmentation-usingDeepLearning

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place the **Micro-Ultrasound Prostate Segmentation Dataset** in the project root, then run:

```bash
python preprocess.py --raw-dir ./Micro_Ultrasound_Prostate_Segmentation_Dataset --output-dir ./data
```

This converts 3D NIfTI volumes to 2D PNG slices (512×512) with an 80/20 train/val split.

### 3. Train the Model

```bash
# Default training (50 epochs)
python train.py

# Custom training
python train.py --epochs 100 --batch-size 4 --lr 1e-4 --data-dir ./data
```

The best model is saved to `files/model.keras`.

### 4. Evaluate

```bash
python evaluate.py
```

Outputs accuracy, F1, IoU, recall, and precision. Visual comparisons (Original | Ground Truth | Prediction) are saved to `results/`.

### 5. Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload an ultrasound image and get instant segmentation results with downloadable masks.

## Docker Deployment

```bash
# Build
docker build -t prostate-seg .

# Run
docker run -p 8501:8501 prostate-seg
```

Then open `http://localhost:8501`.

## Configuration

All settings are centralized in `config.py`. Override via CLI arguments or environment variables:

| Setting | Default | CLI Flag | Env Variable |
|---|---|---|---|
| Data directory | `./data` | `--data-dir` | `PROSTATE_DATA_DIR` |
| Model path | `files/model.keras` | `--model-path` | — |
| Image size | 512×512 | — | — |
| Batch size | 2 | `--batch-size` | — |
| Learning rate | 1e-4 | `--lr` | — |
| Epochs | 50 | `--epochs` | — |

## Results

We evaluated the trained VGG16-UNet architecture on 430 validation micro-ultrasound slices. The final PyTorch model achieved the following performance metrics:

| Metric | Score |
| --- | --- |
| **Accuracy** | `0.9528` (95.28%) |
| **F1 Score (Dice)** | `0.8318` (83.18%) |
| **Jaccard (IoU)** | `0.7842` (78.42%) |
| **Recall** | `0.9206` (92.06%) |
| **Precision** | `0.8539` (85.39%) |

These strong scores—especially the high F1 and Recall—demonstrate the model's robustness at identifying prostate boundaries with high sensitivity, effectively translating expert annotations down to the pixel level.

Evaluation results, alongside generated overlays for every predicted image, are saved in the configured results directory.

## License

This project is for research and educational purposes.
