# NeuroSwift.AI — Progressive ICH Detection
**The Intelligent Edge in Intracranial Diagnostics**

> ⚠ **RESEARCH USE ONLY** — Not a certified medical device. Always consult a qualified radiologist.

---

## Quick Start (Local)

```bash
# 1. Clone / copy the project files into a folder
cd neuroswift/

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **[LINK](https://progressive-ich.streamlit.app/)**

---

## File Structure

```
neuroswift/
├── app.py              # Main Streamlit application (all-in-one)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Loading Your Trained `.pth` Weights

The sidebar contains a **"Load .pth Weights"** uploader. The app auto-detects two checkpoint formats:

```python
# Format 1 — raw state dict
torch.save(model.state_dict(), 'weights.pth')

# Format 2 — full checkpoint (model_state_dict key)
torch.save({
    'epoch': 50,
    'model_state_dict': model.state_dict(),
}, 'checkpoint.pth')
```

### Input Shape Requirements

| Mode    | Tensor Shape   | Description                                              |
|---------|----------------|----------------------------------------------------------|
| single  | (B, 1, 512, 512) | One channel per scan; two independent forward passes   |
| siamese | (B, 2, 512, 512) | Baseline + Current stacked into 2 channels; one pass   |

Make sure your `.pth` file was trained with matching `in_channels` (1 or 2).

---

## Pipeline Overview

```
Upload DICOM pair
      ↓
Load pixels + apply RescaleSlope/Intercept (HU conversion)
      ↓
Brain Window: Level=40, Width=80 HU  →  [0, 1] float
      ↓
Resize & pad to 512×512 (aspect-ratio preserving)
      ↓
Rigid Registration (SimpleITK Euler2D, Mattes MI, 3-level pyramid)
      ↓
U-Net Segmentation (MONAI, single or siamese mode)
      ↓
Area comparison + Progression % calculation
      ↓
Classification: Progressive ICH  /  Stabilized
```

---

## Deployment Options

### Option A — Streamlit Cloud (Free, Fastest)
1. Push `app.py` + `requirements.txt` to a **public** GitHub repo.
2. Go to https://streamlit.io/cloud → "New app" → connect your repo.
3. Set **main file** to `app.py`.

> Note: Streamlit Cloud has a 1 GB memory limit. For full PyTorch + MONAI, use a paid tier or Hugging Face Spaces.

### Option B — Hugging Face Spaces (Free GPU available)
1. Create a new Space: https://huggingface.co/new-space
2. Select **Streamlit** as the SDK.
3. Upload `app.py` and `requirements.txt`.
4. For GPU inference, select a Space with T4 hardware.

### Option C — Docker (Production)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t neuroswift .
docker run -p 8501:8501 neuroswift
```

---

## Configuration (Sidebar)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Inference Mode | single | `single`: one pass per scan. `siamese`: 2-channel stacked |
| Segmentation Threshold | 0.50 | Sigmoid probability cutoff for positive pixels |
| Progression Threshold | 20% | Relative area increase above this → Progressive ICH |
| Weights file | None | Upload `.pth` checkpoint for real segmentation |

---

## Clinical Formula

$$\text{Progression \%} = \left( \frac{A_{\text{current}} - A_{\text{baseline}}}{A_{\text{baseline}}} \right) \times 100$$

Classifications:
- **Progressive ICH**: Progression % > threshold → urgent neurosurgical review
- **Stabilized**: Progression % ≤ threshold → continue observation

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pydicom` | DICOM file parsing |
| `SimpleITK` | Rigid 2D registration |
| `monai` | U-Net architecture |
| `torch` / `torchvision` | Deep learning inference |
| `matplotlib` | Scan rendering with overlay |
| `numpy` / `Pillow` | Array & image utilities |
| `pylibjpeg` | JPEG-compressed DICOM support |

---

*NeuroSwift.AI — Research Prototype — Not for clinical use*
