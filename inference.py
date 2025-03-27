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
import logging
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"Model dtype: {self.model_dtype}")
        
        # Add classification head with matching dtype
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, len(IDX_TO_ANSWER))
        ).to(device, dtype=self.model_dtype)
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
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
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, len(IDX_TO_ANSWER))
            ).to(self.device, dtype=self.model_dtype)
            
            # Initialize weights
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Load and preprocess image
        try:
            pil_image = self.load_image(image_source)
            # Convert PIL Image to tensor and add batch dimension
            image = self.image_transform(pil_image).unsqueeze(0)
            logger.info(f"Image tensor shape: {image.shape}, dtype: {image.dtype}")
            logger.info(f"Image stats - min: {image.min().item():.4f}, max: {image.max().item():.4f}, mean: {image.mean().item():.4f}, std: {image.std().item():.4f}")
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
        logger.info(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        
        # Move inputs and image to device with correct dtypes
        # Keep input_ids as long integers, convert other inputs to model dtype
        inputs = {
            k: v.to(self.device, dtype=self.model_dtype if k != 'input_ids' else torch.long)
            for k, v in inputs.items()
        }
        image = image.to(self.device, dtype=self.model_dtype)
        
        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            try:
                # Get model outputs
                outputs = self.model(
                    **inputs,
                    images=image,
                    output_hidden_states=True,  # Request hidden states
                    return_dict=True  # Ensure we get a dictionary output
                )
                
                # Log model outputs
                logger.info(f"Model outputs keys: {outputs.keys()}")
                if hasattr(outputs, 'hidden_states'):
                    logger.info(f"Number of hidden states: {len(outputs.hidden_states)}")
                    logger.info(f"Hidden states shapes: {[h.shape for h in outputs.hidden_states]}")
                    
                    # Check for NaN in hidden states
                    for i, hidden_state in enumerate(outputs.hidden_states):
                        if torch.isnan(hidden_state).any():
                            logger.warning(f"NaN found in hidden state {i}")
                            logger.warning(f"Hidden state {i} stats - min: {hidden_state.min().item():.4f}, max: {hidden_state.max().item():.4f}, mean: {hidden_state.mean().item():.4f}, std: {hidden_state.std().item():.4f}")
                
                # Use an earlier layer's hidden states (before NaN propagation)
                # We'll use layer 6 since NaN starts at layer 7
                hidden_states = outputs.hidden_states[6]  # Get hidden states from layer 6
                logger.info(f"Selected hidden state shape: {hidden_states.shape}")
                
                # Use the last token's representation for classification
                last_hidden_state = hidden_states[:, -1, :]
                logger.info(f"Last token hidden state shape: {last_hidden_state.shape}")
                
                # Check for NaN in last hidden state
                if torch.isnan(last_hidden_state).any():
                    logger.warning("NaN found in last hidden state")
                    # Try to handle NaN values
                    last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0)
                
                logger.info(f"Last token hidden state stats - min: {last_hidden_state.min().item():.4f}, max: {last_hidden_state.max().item():.4f}, mean: {last_hidden_state.mean().item():.4f}, std: {last_hidden_state.std().item():.4f}")
                
                # Get logits and probabilities using our classifier
                logits = self.classifier(last_hidden_state)
                logger.info(f"Logits shape: {logits.shape}")
                logger.info(f"Logits values: {logits}")
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    logger.warning("NaN found in logits")
                    # Try to handle NaN values
                    logits = torch.nan_to_num(logits, nan=0.0)
                
                logger.info(f"Logits stats - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                
                # Apply temperature scaling to control confidence
                temperature = 0.5  # Lower temperature for more confident predictions
                scaled_logits = logits / temperature
                logger.info(f"Scaled logits: {scaled_logits}")
                
                # Use log_softmax for better numerical stability
                log_probs = F.log_softmax(scaled_logits, dim=1)
                logger.info(f"Log probabilities: {log_probs}")
                
                probabilities = torch.exp(log_probs)[0]
                logger.info(f"Probabilities: {probabilities}")
                
                # Get predicted answer
                pred_idx = torch.argmax(probabilities).item()
                predicted_answer = self.idx_to_answer[pred_idx]
                
                # Get confidence scores for all options
                confidence_scores = {
                    answer: float(probabilities[idx].item())  # Convert to float to avoid any dtype issues
                    for idx, answer in self.idx_to_answer.items()
                }
                logger.info(f"Confidence scores: {confidence_scores}")
                
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                return {
                    'error': f"Inference error: {str(e)}",
                    'predicted_answer': None,
                    'confidence_scores': None
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