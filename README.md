# Exam Question Answer Prediction

This project implements a machine learning system for automatically predicting answers to multiple-choice exam questions from images. It uses Google's Gemma model family to analyze question images and predict the correct answer choice (A, B, C, D, or E).

## Features

- Support for Google's Gemma models:
  - `google/gemma-3-4b-it` (instruction-tuned)
  - `google/gemma-3-4b-pt` (pretrained)
- Integration with the Rocktim/EXAMS-V dataset
- Multiple operation modes:
  - Training with validation
  - Model evaluation with detailed metrics
  - Single question inference
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confidence scores for each answer choice

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

The project provides a command-line interface through `main.py` with three main operation modes:

### Training

Train the model on the EXAMS-V dataset:

```bash
python main.py --mode train --model gemma-it \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3
```

Optional arguments:
- `--checkpoint`: Path to a checkpoint to resume training

### Evaluation

Evaluate the model's performance on the test set:

```bash
python main.py --mode test --model gemma-it \
    --batch_size 8 \
    --checkpoint checkpoints/gemma-it_final.pt
```

This will output:
- Average loss
- Accuracy
- Precision
- Recall
- F1 Score
- Sample predictions

### Inference

Predict the answer for a single exam question image:

```bash
python main.py --mode infer --model gemma-it \
    --image path/to/question.jpg \
    --checkpoint checkpoints/gemma-it_final.pt
```

This will output:
- Predicted answer (A, B, C, D, or E)
- Confidence scores for each option

## Model Details

The system uses a Gemma model with an added classification head to predict multiple-choice answers:

1. The image and a prompt are processed by the model
2. The model's output is passed through a classification layer
3. Softmax is applied to get probabilities for each answer choice
4. The answer with the highest probability is selected

## Dataset

The project uses the Rocktim/EXAMS-V dataset, which contains:
- Multiple-choice exam questions as images
- Correct answer keys (A, B, C, D, E)
- Subject information (e.g., Biology, Natural Science)
- Grade levels

## Model Checkpoints

Trained models are automatically saved to the `checkpoints/` directory:
- Best performing models during training (based on validation accuracy)
- Final model after training completion

## Requirements

- Python 3.8+
- PyTorch 2.1+
- Transformers 4.37+
- Other dependencies listed in requirements.txt

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 