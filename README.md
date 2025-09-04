# DeepSeek-from-Scratch (Educational Implementation)

## Overview
This repository contains a **from-scratch implementation of a Large Language Model (LLM)** 
It is built entirely in **PyTorch**, with a focus on clarity, modularity, and extensibility for rapid experimentation with novel LLM architectures.

The model incorporates modern Transformer advancements, including:
- **Multi-Head Attention** with **Multi-head Latent Attention (MLA)**
- **Rotary Position Embeddings (RoPE)**
- **Mixture of Experts (MoE)** with top-2 routing
- **SwiGLU** activation
- **KV cache** for efficient autoregressive inference
- **Weight tying** for input/output embeddings

---

## Features
- **Educational clarity:** All components are implemented from scratch with detailed comments.
- **Advanced attention mechanisms:** MLA improves efficiency by reducing key/value head dimensionality while preserving performance.
- **MoE layers:** Enable sparse computation and specialization of experts.
- **RoPE embeddings:** Enhance positional encoding for extrapolation.
- **Demo-ready:** Includes a small training loop and generation script.

---

## File Structure
deepseek_from_scratch.py # Single-file implementation of the model and training loop
requirements.txt # Dependencies
README.md # Project documentation



---

## Installation
```bash
git clone https://github.com/<your-username>/deepseek-from-scratch.git
cd deepseek-from-scratch
pip install -r requirements.txt

Usage

Run the included demo (tiny dataset training + text generation):
python deepseek_from_scratch.py --demo

without demo

python deepseek_from_scratch.py

Requirements

Python 3.9+

PyTorch >= 2.0

Install dependencies:
pip install torch


Disclaimer

This implementation is for *educational purposes* only.
It is *not* the official DeepSeek code and may differ in low-level details.




