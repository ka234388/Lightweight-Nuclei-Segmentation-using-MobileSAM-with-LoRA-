
# CAP 5516 Assignment 3 – MobileSAM with LoRA for Nuclei Instance Segmentation

This repository contains the implementation of **parameter-efficient fine-tuning** of the **MobileSAM** model using **LoRA** (Low-Rank Adaptation) for **nuclei instance segmentation** on the **NuInsSeg dataset**. The project is part of the Medical Image Computing course (Spring 2025) at UCF.

---

## 📂 Dataset

- **Name:** NuInsSeg  
- **Description:** A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images.
- **Source:** [NuInsSeg Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg)  
- **Format:** PNG image files and TIFF mask files grouped by organ.

---

## Model Overview

- **Backbone:** MobileSAM (TinyViT-based lightweight SAM)
- **Fine-Tuning Method:** LoRA (applied to all linear layers in image encoder)
- **Frozen Components:** Prompt encoder, original SAM weights (except decoder and LoRA layers)
- **Trainable Parameters:** ~4.18M out of 10.25M (~40.78%)

---

## Code Features

- **Dataset Loader:** Custom `NuInsSegDatasetFunction` class for resizing and normalization.
- **LoRA Injection:** Custom `LoRALinearFunction` applied selectively to all `nn.Linear` layers in the SAM encoder.
- **Training Loop:** Includes loss computation using `BCEWithLogitsLoss`, forward/backward passes, and optimization.
- **Post-processing:** Watershed algorithm used for instance separation.
- **Evaluation Metrics:** Dice Score, Aggregated Jaccard Index (AJI), and Panoptic Quality (PQ).

---

## 📈 Results Summary

| Metric      | 5 Epochs Avg | 10 Epochs Avg |
|-------------|--------------|---------------|
| Dice Score  | 0.6122       | 0.6611        |
| AJI Score   | 0.3660       | 0.4114        |
| PQ Score    | 0.3014       | 0.3959        |

10 epochs show significant improvement in all metrics, especially PQ.

---

##  Visual Comparison

- Includes visual outputs for:
  - Original Image
  - Ground Truth Mask
  - Predicted Mask
  - Watershed Instance Output

---

##  How to Run

1. **Download Dataset:**
   - Available on [Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg)

2. **Clone Repository & Install Requirements:**
   ```bash
   git clone https://github.com/<your-username>/mobile-sam-nuinsseg-lora.git
   cd mobile-sam-nuinsseg-lora
   pip install -r requirements.txt
   ```

3. **Run Notebook:**
   - Open `CAP_5516_Assignment3.ipynb` in Kaggle or Jupyter and execute all cells.

---

## 📊 Evaluation Strategy

- **5-Fold Cross Validation**
- **Metrics tracked per fold**
- Final average used for performance reporting

---

##  Dependencies

- PyTorch
- OpenCV
- scikit-image
- tifffile
- matplotlib
- tqdm
- kagglehub

---

## Reference Papers

- [NuInsSeg Dataset](https://arxiv.org/abs/2308.01760)
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643)
- [MobileSAM](https://arxiv.org/abs/2306.14289)
- [LoRA for Efficient Tuning](https://arxiv.org/abs/2106.09685)

---

## Author

- **Name:** *Karthika Ramasamy*
- **Course:** CAP 5516 – Medical Image Computing
- **Instructor:** Dr. Chen Chen
- **Semester:** Spring 2025
