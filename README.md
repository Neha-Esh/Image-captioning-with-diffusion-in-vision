# Image Captioning using Diffusion Models in Vision

**Author**: Neha Eshwaragari  
**Course**: Artificial Intelligence Systems  
**Institution**: University of Florida  

## Project Overview

This project implements a deep learning pipeline that combines computer vision and natural language processing to:
1. Automatically generate descriptive captions for input images.
2. Optionally reconstruct or generate images conditioned on those captions.

It integrates a CNN-based image encoder (ResNet-50) with a Transformer-based language decoder (GPT-2), utilizing transfer learning to understand and describe visual content. A Gradio interface allows real-time interaction with the system.

---

## Methodology

### Libraries Used
- `torch`, `torchvision` — deep learning models and feature extraction
- `transformers` (Hugging Face) — pre-trained GPT-2 language model
- `NLTK` — BLEU, METEOR evaluation metrics
- `scikit-image` — PSNR, SSIM image similarity metrics
- `Gradio` — interactive UI
- `NumPy`, `Matplotlib`, `PIL` — preprocessing, visualization

### Architecture
- **Image Encoder**: ResNet-50 (pre-trained, truncated before final layer) → extracts 2048-dimensional features.
- **Language Decoder**: GPT-2 (pre-trained) → generates captions based on image features.
- **Fusion Layer**: A learned linear transformation aligns image embeddings with GPT-2’s input space.

### Captioning Strategies
- **Greedy Decoding** for deterministic outputs
- **Top-k Sampling** for diverse, creative captions

---

## Evaluation Metrics

| Type | Metrics |
|------|---------|
| **Textual** | BLEU-1, BLEU-4, METEOR |
| **Image-based** | PSNR, SSIM |

---

## Results

### Quantitative
- **BLEU-1**: 61.3%
- **BLEU-4**: 28.5%
- **METEOR**: 25.7%
- **CIDEr**: 75.1

### Qualitative
- Captions were generally fluent and semantically accurate.
- Some struggles with images containing multiple or subtle actions.

---

## Training Details

- **Dataset**: Flickr8K
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5 (linear decay)
- **Batch Size**: 32
- **Epochs**: 5
- **Loss**: Cross-Entropy

> Training took ~3 hours on Jupyter due to memory limits.

---

##  Demonstration

A video titled `Caption generating.mp4` demonstrates the system’s real-time functionality using Gradio:
- Upload an image
- Extract features via ResNet-50
- Generate and display caption using GPT-2

---

## Running the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/project3-captioning.git
   cd project3-captioning
    ```
2. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or start the Gradio UI:
   ``` bash
   jupyter notebook Project3.ipynb
   # or to launch the app
   python app.py
    ```
---

## Future work

- Integrate attention mechanisms for finer detail recognition.

- Experiment with larger or more efficient language models.

- Enable scalable real-time captioning.

- Improve generalization on complex scenes with multiple objects.

---

## References 
- [1] He et al., “Deep Residual Learning for Image Recognition” (CVPR, 2016)

- [2] Radford et al., “Language Models are Unsupervised Multitask Learners” (OpenAI, 2019)

- [3] Xu et al., “Show, Attend and Tell” (ICML, 2015)

- [4] Ho et al., “Denoising Diffusion Probabilistic Models” (NeurIPS, 2020)

- [5] Chen et al., “Imagen: Text-to-Image Diffusion Models” (arXiv, 2022)

---
## Author
Neha Eshwaragari

University of Florida

neha.eshwaragari@ufl.edu


