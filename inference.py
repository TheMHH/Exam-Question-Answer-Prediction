import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Union
from model_loader import ModelLoader
import requests
from io import BytesIO
import os
from constants import EXAM_QUESTION_CHAT_TEMPLATE, EXTRACT_ANSWER_PROMPT

class ExamInferencer:
    def __init__(self, model_type: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the inferencer using ModelLoader.
        
        Args:
            model_type (str): Type of model to use
            device (str): Device to run inference on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.device = device
        
        # Load model and tokenizer using ModelLoader
        self.model_loader = ModelLoader(model_type, device)
        self.model, self.processor = self.model_loader.load_base_model()
    
    def load_image(self, image_source: str) -> Image.Image:
        """Load an image from a URL or local path."""
        try:
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                image = Image.open(image_source)
            
            return image.convert("RGB")
        except requests.RequestException as e:
            raise ValueError(f"Error downloading image from URL: {e}")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    def predict_answer(self, image_source: str) -> Dict:
        """Generate an answer using both image and text input."""
        pil_image = self.load_image(image_source)
        
        
        prompt = self.processor.apply_chat_template(
            EXAM_QUESTION_CHAT_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device, torch.bfloat16)
                
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1000,
            )
            
        full_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        parts = full_text.split("model")
        model_response = parts[-1].strip() if len(parts) > 1 else full_text.strip()  
            
        return {
            'generated_text': model_response,
            'error': None
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate an answer for an exam question with image context")
    parser.add_argument("--model", required=True, help="Model type to use")
    parser.add_argument("--image", required=True, help="URL or path to the input image")
    
    args = parser.parse_args()
    
    inferencer = ExamInferencer(args.model)
    result = inferencer.predict_answer(args.image)
    
    if result['error']:
        print(f"\nError: {result['error']}")
    else:
        print("\nImage Context:")
        print(f"Format: {result['image_metadata']['format']}")
        print(f"Dimensions: {result['image_metadata']['width']}x{result['image_metadata']['height']}")
        print(f"\nGenerated Answer: {result['predicted_answer']}")
        print(f"\nFull Generation:\n{result['generated_text']}")


if __name__ == "__main__":
    main()
