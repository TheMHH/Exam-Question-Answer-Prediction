from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple, Optional

class ModelLoader:
    MODEL_MAPPING = {
        "gemma-it": "google/gemma-3-4b-it",
        "gemma-pt": "google/gemma-3-4b-pt"
    }

    def __init__(self, model_type: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the model loader.
        
        Args:
            model_type (str): Type of model to load ("gemma-it" or "gemma-pt")
            device (str): Device to load the model on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, checkpoint_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.
        
        Args:
            checkpoint_path (Optional[str]): Path to a saved checkpoint
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
        """
        model_name = self.MODEL_MAPPING[self.model_type]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        if checkpoint_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
        
        return self.model, self.tokenizer

    def save_model(self, save_path: str) -> None:
        """
        Save the current model and tokenizer.
        
        Args:
            save_path (str): Path to save the model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path) 