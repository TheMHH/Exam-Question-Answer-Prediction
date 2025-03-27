import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Dict, Union
from model_loader import ModelLoader
import requests
from io import BytesIO
import os
from constants import EXAM_QUESTION_PROMPT, IDX_TO_ANSWER

class ExamInferencer:
    def __init__(
        self,
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the inferencer.
        
        Args:
            model_type (str): Type of model to use ("gemma-it" or "gemma-pt")
            device (str): Device to run inference on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.device = device
        
        # Initialize model loader
        self.model_loader = ModelLoader(model_type, device)
        self.model, self.tokenizer = self.model_loader.load_model()
        
        # Get model dtype
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Add classification head with matching dtype
        hidden_size = self.model.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, len(IDX_TO_ANSWER)).to(device, dtype=self.model_dtype)
        
        # Answer mapping
        self.idx_to_answer = IDX_TO_ANSWER
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to common size
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(            # Normalize for model
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, image_source: str) -> Image.Image:
        """
        Load and preprocess an image from either a URL or local path.
        
        Args:
            image_source (str): URL or path to the image file
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # Download image from URL
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()  # Raise an error for bad status codes
                image = Image.open(BytesIO(response.content))
            else:
                # Load from local path
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                image = Image.open(image_source)
            
            return image.convert('RGB')
        except requests.RequestException as e:
            raise ValueError(f"Error downloading image from URL: {e}")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def predict_answer(
        self,
        image_source: str,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Predict the answer for an exam question image.
        
        Args:
            image_source (str): URL or path to the input image
            checkpoint_path (Optional[str]): Path to a checkpoint to use
            
        Returns:
            Dict: Dictionary containing predicted answer and confidence scores
        """
        if checkpoint_path:
            self.model, self.tokenizer = self.model_loader.load_model(checkpoint_path)
            # Get model dtype
            self.model_dtype = next(self.model.parameters()).dtype
            # Recreate classifier after loading checkpoint with matching dtype
            hidden_size = self.model.config.hidden_size
            self.classifier = torch.nn.Linear(hidden_size, len(IDX_TO_ANSWER)).to(self.device, dtype=self.model_dtype)
        
        # Load and preprocess image
        try:
            pil_image = self.load_image(image_source)
            # Convert PIL Image to tensor and add batch dimension
            image = self.image_transform(pil_image).unsqueeze(0)
        except Exception as e:
            return {
                'error': str(e),
                'predicted_answer': None,
                'confidence_scores': None
            }
        
        # Prepare inputs
        inputs = self.tokenizer(
            EXAM_QUESTION_PROMPT,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True
        )
        
        # Move inputs and image to device with correct dtype
        inputs = {k: v.to(self.device, dtype=self.model_dtype) for k, v in inputs.items()}
        image = image.to(self.device, dtype=self.model_dtype)
        
        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(
                **inputs,
                images=image,
                output_hidden_states=True  # Request hidden states
            )
            
            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]  # Get last layer's hidden states
            # Use the last token's representation for classification
            last_hidden_state = hidden_states[:, -1, :]
            
            # Get logits and probabilities using our classifier
            logits = self.classifier(last_hidden_state)
            probabilities = F.softmax(logits, dim=1)[0]
            
            # Get predicted answer
            pred_idx = torch.argmax(probabilities).item()
            predicted_answer = self.idx_to_answer[pred_idx]
            
            # Get confidence scores for all options
            confidence_scores = {
                answer: probabilities[idx].item()
                for idx, answer in self.idx_to_answer.items()
            }
        
        return {
            'predicted_answer': predicted_answer,
            'confidence_scores': confidence_scores,
            'source_type': 'url' if image_source.startswith(('http://', 'https://')) else 'local'
        }

def main():
    """
    Example usage of the ExamInferencer class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict answer for exam question image")
    parser.add_argument("--model", choices=["gemma-it", "gemma-pt"], required=True,
                      help="Model type to use")
    parser.add_argument("--image", required=True,
                      help="URL or path to the input image")
    parser.add_argument("--checkpoint", help="Path to a model checkpoint")
    
    args = parser.parse_args()
    
    # Initialize inferencer
    inferencer = ExamInferencer(args.model)
    
    # Predict answer
    result = inferencer.predict_answer(
        args.image,
        checkpoint_path=args.checkpoint
    )
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
    else:
        print("\nPrediction Results:")
        print(f"Image Source Type: {result['source_type']}")
        print(f"Predicted Answer: {result['predicted_answer']}")
        print("\nConfidence Scores:")
        for answer, score in result['confidence_scores'].items():
            print(f"{answer}: {score:.4f}")

if __name__ == "__main__":
    main() 