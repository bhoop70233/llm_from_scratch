# LLM from Scratch

This repository contains implementations of Large Language Models (LLMs) from scratch, following the concepts and techniques used in modern transformer-based models.

## Project Structure

- `ch-2.ipynb` - Chapter 2: Data Loading and Preprocessing
- `corrected_gpt_dataset.py` - Corrected implementation of GPT dataset class
- `the-verdict.txt` - Sample text data for training

## Features

- Custom PyTorch dataset implementation for text tokenization
- GPT-style data loading with sliding window approach
- Tiktoken tokenizer integration
- Configurable sequence length and stride parameters

## Requirements

```bash
pip install torch tiktoken numpy
```

## Usage

1. Open `ch-2.ipynb` in Jupyter Notebook
2. Run the cells to see the data loading implementation
3. The notebook demonstrates how to create custom datasets for language modeling

## Dataset Class

The `GPTDatasetV1` class implements:
- Text tokenization using tiktoken
- Sliding window approach for sequence generation
- Input-target pair creation for language modeling
- PyTorch Dataset compatibility

## License

This project is for educational purposes.
