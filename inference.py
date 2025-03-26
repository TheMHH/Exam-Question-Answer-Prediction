import torch
from PIL import Image
from typing import Optional
from model_loader import ModelLoader

class Inferencer:
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

    def generate_text(
        self,
        image_path: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.7,
        checkpoint_path: Optional[str] = None
    ) -> str:
        """
        Generate text from an image.
        
        Args:
            image_path (str): Path to the input image
            max_length (int): Maximum length of generated text
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for text generation
            checkpoint_path (Optional[str]): Path to a checkpoint to use
            
        Returns:
            str: Generated text
        """
        if checkpoint_path:
            self.model, self.tokenizer = self.model_loader.load_model(checkpoint_path)
        
        # Load and preprocess image
        image = self.load_image(image_path)
        
        # Prepare inputs
        inputs = self.tokenizer(
            "",
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image = image.to(self.device)
        
        # Generate text
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                images=image,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True
            )
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text

def main():
    """
    Example usage of the Inferencer class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text from an image")
    parser.add_argument("--model", choices=["gemma-it", "gemma-pt"], required=True,
                      help="Model type to use")
    parser.add_argument("--image", required=True,
                      help="Path to the input image")
    parser.add_argument("--checkpoint", help="Path to a model checkpoint")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=4,
                      help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for text generation")
    
    args = parser.parse_args()
    
    # Initialize inferencer
    inferencer = Inferencer(args.model)
    
    # Generate text
    generated_text = inferencer.generate_text(
        args.image,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint
    )
    
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main() 