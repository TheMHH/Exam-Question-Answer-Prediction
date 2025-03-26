import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict
from model_loader import ModelLoader

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
        
        # Answer mapping
        self.idx_to_answer = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'
        }

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Image.Image: Preprocessed image
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def predict_answer(
        self,
        image_path: str,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Predict the answer for an exam question image.
        
        Args:
            image_path (str): Path to the input image
            checkpoint_path (Optional[str]): Path to a checkpoint to use
            
        Returns:
            Dict: Dictionary containing predicted answer and confidence scores
        """
        if checkpoint_path:
            self.model, self.tokenizer = self.model_loader.load_model(checkpoint_path)
        
        # Load and preprocess image
        image = self.load_image(image_path)
        
        # Create input prompt
        prompt = "Look at this exam question image and select the correct answer choice (A, B, C, D, or E):"
        
        # Prepare inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image = image.to(self.device)
        
        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                images=image
            )
            
            # Get logits and probabilities
            logits = self.model.classifier(outputs.last_hidden_state[:, 0, :])
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
            'confidence_scores': confidence_scores
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
                      help="Path to the input image")
    parser.add_argument("--checkpoint", help="Path to a model checkpoint")
    
    args = parser.parse_args()
    
    # Initialize inferencer
    inferencer = ExamInferencer(args.model)
    
    # Predict answer
    result = inferencer.predict_answer(
        args.image,
        checkpoint_path=args.checkpoint
    )
    
    print("\nPrediction Results:")
    print(f"Predicted Answer: {result['predicted_answer']}")
    print("\nConfidence Scores:")
    for answer, score in result['confidence_scores'].items():
        print(f"{answer}: {score:.4f}")

if __name__ == "__main__":
    main() 