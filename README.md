# Image Captioning Project

This repository contains two implementations of image captioning using deep learning:
1. A **Streamlit web application** powered by the pre-trained BLIP model for real-time image captioning.
2. A **Jupyter Notebook** implementing a custom CNN-Transformer model trained on the COCO dataset for image captioning.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)

## Project Overview

This project demonstrates two approaches to image captioning:
- The **Streamlit app** uses the Salesforce BLIP model (`Salesforce/blip-image-captioning-base`) to generate captions for uploaded images in a user-friendly web interface.
- The **Jupyter Notebook** builds a custom CNN-Transformer model, preprocesses the COCO dataset, trains the model, and generates captions. It also includes functionality to save model components.

The project is suitable for researchers, developers, and enthusiasts interested in computer vision and natural language processing.

## Features

- **Streamlit App (BLIP)**:
  - Upload images (JPG, JPEG, PNG) to generate captions instantly.
  - Intuitive web interface built with Streamlit.
  - Leverages the pre-trained BLIP model for high-quality captions.
- **Transformer Model (Notebook)**:
  - Preprocesses captions from the COCO dataset.
  - Implements a CNN-Transformer architecture for end-to-end image captioning.
  - Supports saving model components (CNN, encoder, decoder weights, and tokenizer vocabulary).
  - Includes a sample caption generation for a test image.

