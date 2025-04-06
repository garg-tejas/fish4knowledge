# Fish4Knowledge Dataset Processing Tool

This repository contains tools for processing and utilizing the Fish4Knowledge dataset for computer vision tasks, primarily focused on fish segmentation through image-mask pairs.

## Overview

The Fish4Knowledge project provides utilities to:

1. Split the raw Fish4Knowledge dataset into train, validation, and test sets
2. Load and process image-mask pairs for machine learning models
3. Train a fish segmentation and classification model
4. Visualize and interact with the trained model

## Directory Structure

The expected dataset structure for input:

```
data/FishDataset/
├── fish_1/
│   ├── fish_1_001.png
│   ├── fish_1_002.png
│   └── ...
├── mask_1/
│   ├── mask_1_001.png
│   ├── mask_1_002.png
│   └── ...
├── fish_2/
│   └── ...
├── mask_2/
│   └── ...
└── ...
```

After processing, the output structure will be:

```
data/
├── train/
│   ├── 01/
│   │   ├── images/
│   │   └── masks/
│   ├── 02/
│   │   ├── images/
│   │   └── masks/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/garg-tejas/fish4knowledge.git
cd fish4knowledge
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Splitting

To split your dataset into train, validation, and test sets:

```bash
python data-splitting.py
```

The script uses these default parameters:

- Source directory: FishDataset
- Output directory: data
- Split ratios: 70% train, 20% validation, 10% test

### Dataset Loading

The fish_dataset.py module provides functionality to load the processed dataset for training models.

### Model Training

To train the segmentation and classification model:

```bash
python train.py
```

The training script:

- Uses PyTorch with CUDA acceleration (if available)
- Performs both segmentation (mask prediction) and classification tasks
- Saves checkpoints in the checkpoints directory
- Logs metrics with TensorBoard in the fish_experiment directory

### Interactive Application

The project includes an interactive application to visualize and test the trained model:

```bash
streamlit run app.py
```

This provides a user interface for uploading images and viewing segmentation and classification results.

## Required Data

The Fish4Knowledge dataset can be obtained from [official sources](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/). Place the downloaded dataset in the FishDataset directory before running the splitting script.

## Notes

- The data splitting process matches image-mask pairs based on their file naming pattern
- Images and masks are expected to be in PNG format
- The random seed is set to 42 for reproducible splits
- The model performs both segmentation and species classification
