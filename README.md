# Transformers Architecture from Scratch in Python

## Overview
This project implements the Transformers architecture from scratch in Python. It includes the following components:
- Self-Attention
- Causal Attention
- Cross Attention
- GPT (Generative Pre-trained Transformer) Model
- BERT (Bidirectional Encoder Representations from Transformers) Model

## Components
### Self-Attention
Self-attention mechanism computes the attention score for each position in the input sequence.

### Causal Attention
Causal attention is a variant of self-attention where each position can only attend to previous positions.

### Cross Attention
Cross attention computes the attention score between two sequences, typically used in tasks such as machine translation.

### GPT Model
The Generative Pre-trained Transformer (GPT) model is a variant of the Transformer architecture used for various natural language processing tasks.

### BERT Model
BERT (Bidirectional Encoder Representations from Transformers) is a Transformer-based machine learning model for natural language processing tasks.

## Implementation
The project is implemented in Python using NumPy for numerical computations.

### Files
- `attention.py`: Implementation of self-attention mechanism.
- `attention.py`: Implementation of causal attention mechanism.
- `attention.py`: Implementation of cross attention mechanism.
- `GPT.py`: Implementation of the GPT model.
- `bert.py`: Implementation of the BERT model.

## Dependencies
- Python 3.10.12
- Pytorch

## Acknowledgments
This project is inspired by the Transformers architecture introduced in the paper "Attention Is All You Need" by Vaswani et al.


