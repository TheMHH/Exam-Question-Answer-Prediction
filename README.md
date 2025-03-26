# Image-Text-to-Text Model Project

This project implements a modular framework for working with image-text-to-text models using Hugging Face's transformers and datasets libraries. It supports model loading, fine-tuning, evaluation, and inference for the Gemma model family.

## Features

- Support for Google's Gemma models (3-4B instruction-tuned and pretrained variants)
- Integration with Rocktim/EXAMS-V dataset
- Modular architecture with separate components for:
  - Model loading and management
  - Dataset handling and preprocessing
  - Model fine-tuning
  - Model evaluation
  - Inference on custom images

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── requirements.txt
├── README.md
├── main.py
├── model_loader.py
├── data_loader.py
├── train.py
├── test.py
├── inference.py
└── checkpoints/
```

## Usage

The project provides a command-line interface through `main.py` with the following options:

```bash
python main.py [--mode {test,train,infer}] [--model {gemma-it,gemma-pt}] [--checkpoint PATH] [--image PATH]
```

### Arguments:

- `--mode`: Select the operation mode
  - `test`: Evaluate model on test dataset
  - `train`: Fine-tune the model
  - `infer`: Run inference on a custom image
- `--model`: Choose the model variant
  - `gemma-it`: Google's Gemma 3-4B instruction-tuned model
  - `gemma-pt`: Google's Gemma 3-4B pretrained model
- `--checkpoint`: Path to a saved model checkpoint (optional)
- `--image`: Path to an image file for inference mode

### Examples:

1. Evaluate the instruction-tuned model:
```bash
python main.py --mode test --model gemma-it
```

2. Fine-tune the pretrained model:
```bash
python main.py --mode train --model gemma-pt
```

3. Run inference on a custom image:
```bash
python main.py --mode infer --model gemma-it --image path/to/image.jpg
```

## Model Checkpoints

Fine-tuned models are automatically saved to the `checkpoints/` directory. You can load these checkpoints for further training or inference using the `--checkpoint` argument.

## License

[Your chosen license] 