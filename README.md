# AI Guitar Classifier Project 🎸

This repository contains the Jupyter/Colab notebook and related materials for a project focused on fine-tuning a deep learning vision model to classify different types of electric guitars.

## Project Overview

The core of this project involves taking a pre-trained Convolutional Neural Network (CNN), specifically ResNet architectures, and fine-tuning it on a custom dataset of electric guitar images. The goal was to accurately distinguish between several iconic guitar models.

The accompanying notebook (`Ada447Midterm.ipynb`) details the entire process, including:
*   Data collection, inspection, and preprocessing (including image augmentations and presizing).
*   Setting up the data pipeline using the fastai library.
*   Training a benchmark model using `fine_tune`.
*   Implementing advanced fine-tuning techniques such as:
    *   Using the Learning Rate Finder.
    *   Performing a two-stage transfer learning process (training the head, then unfreezing and training the full model).
    *   Applying discriminative learning rates.
*   Evaluating model performance using metrics like accuracy, top losses, and confusion matrices.
*   Experiments with different model capacities (ResNet34 vs. ResNet50).

## Key Technologies Used
*   Python
*   Fastai & PyTorch
*   Matplotlib (for visualizations)
*   Gradio (for the web application interface)
*   Google Colab (for training)
*   Hugging Face Spaces (for deployment)

## Dive Deeper

📄 **Medium Article: "Six-Stringed Fine-Tuning Trauma"**
For a detailed walkthrough of the project, the decisions made, and how each topic from the project requirements was addressed during the fine-tuning process, please read my blog post:
[Six-Stringed Fine-Tuning Trauma](https://medium.com/@baypnar/six-stringed-fine-tuning-trauma-601239cb3284)

🚀 **Try the Live Demo on Hugging Face Spaces!**
Want to see the model in action? Upload an image of an electric guitar and see if the AI can identify it:
[Guitar Identifier - Hugging Face Space](https://huggingface.co/spaces/yux3l/guitar-identifier)

## Notebook
The primary Jupyter/Colab notebook containing all the code for data processing, model training, and evaluation can be found here:
*   [`Ada447Midterm.ipynb`](./Ada447Midterm.ipynb)

## Dataset
The initial dataset was sourced from Roboflow Universe ("guitars-classification" by felipe-zx8mw) and was subsequently augmented and curated for this project. The version used in the notebook consists of 6 guitar classes after removing the 'SG' type, it is currently hosted privately on Google Drive, however upon request, I can grant access.

