# Verified Reversible Data Hiding Framework

This repository contains the complete implementation of a feasibility-verified reversible data hiding (RDH) framework using prediction-error expansion (PEE), two-stage checkerboard embedding, and automatic threshold verification.

## Main Features

* Two-stage checkerboard embedding
* Directional and interpolation-based prediction
* Prediction-error expansion mapping
* Header embedding and extraction
* CRC verification
* Bit-error-rate validation
* Exact cover-image recovery check
* Automatic threshold search
* PSNR, SSIM, MSE, entropy, and runtime reporting
* CSV generation for full and summary results

## Folder Structure

```text
Verified-RDH-Full/
│
├── main.py
├── README.md
├── requirements.txt
├── input/
│   ├── images/
│   └── payloads/
└── output/
    ├── stego\_images/
    ├── cover\_images/
    └── csv\_results/
```

## Installation

```bash
pip install -r requirements.txt
```

## Input Files

Place grayscale test images in:

```text
input/images/
```

Place binary payload files in:

```text
input/payloads/
```

The default image and payload filenames are defined in `IMAGE\_LIST` and `PAYLOAD\_LIST` inside `main.py`. These lists may be modified as needed, or left empty to automatically process all supported files in the input directories.

## Run

```bash
python main.py
```

## Output

The program saves generated stego images, cover-image copies, and CSV result files in the `output/` directory.

