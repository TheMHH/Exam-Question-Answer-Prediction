from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from constants import IDX_TO_ANSWER

class ModelLoader:
    def __init__(
        self,
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the model loader.
        
        Args:
            model_type (str): Type of model to use ("gemma-it" or "gemma-pt")
            device (str): Device to run inference on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.device = device
        
        # Model configurations
        self.model_configs = {
            "gemma-it": {
                "name": "google/gemma-2b-it",
                "max_length": 2048,
                "hidden_size": 2560
            },
            "gemma-pt": {
                "name": "google/gemma-2b",
                "max_length": 2048,
                "hidden_size": 2560
            }
        }
        
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.config = self.model_configs[model_type]
        
        # Create classifier architecture
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config["hidden_size"]),
            nn.Linear(self.config["hidden_size"], self.config["hidden_size"] // 2),
            nn.LayerNorm(self.config["hidden_size"] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config["hidden_size"] // 2, self.config["hidden_size"] // 4),
            nn.LayerNorm(self.config["hidden_size"] // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config["hidden_size"] // 4, len(IDX_TO_ANSWER))
        )

    def load_model(
        self,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Load the model and tokenizer.
        
        Args:
            checkpoint_path (Optional[str]): Path to a model checkpoint
            
        Returns:
            Tuple[nn.Module, nn.Module]: Model and tokenizer
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["name"],
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config["name"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Move model to device
        model = model.to(self.device)
        
        # Get model dtype
        model_dtype = next(model.parameters()).dtype
        
        # Move classifier to device with matching dtype
        classifier = self.classifier.to(self.device, dtype=model_dtype)
        
        # Initialize classifier weights
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Combine model and classifier
        class CombinedModel(nn.Module):
            def __init__(self, base_model, classifier):
                super().__init__()
                self.base_model = base_model
                self.classifier = classifier
            
            def forward(self, **inputs):
                outputs = self.base_model(**inputs)
                if hasattr(outputs, 'hidden_states'):
                    # Use layer 6's hidden states (before NaN propagation)
                    hidden_states = outputs.hidden_states[6]
                    last_hidden_state = hidden_states[:, -1, :]
                    
                    # Handle NaN values
                    if torch.isnan(last_hidden_state).any():
                        last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0)
                    
                    # Get logits from classifier
                    logits = self.classifier(last_hidden_state)
                    
                    # Handle NaN values in logits
                    if torch.isnan(logits).any():
                        logits = torch.nan_to_num(logits, nan=0.0)
                    
                    # Apply temperature scaling
                    temperature = 0.5
                    scaled_logits = logits / temperature
                    
                    # Get probabilities
                    log_probs = F.log_softmax(scaled_logits, dim=1)
                    probabilities = torch.exp(log_probs)
                    
                    # Add probabilities to outputs
                    outputs.probabilities = probabilities
                
                return outputs
        
        # Create combined model
        combined_model = CombinedModel(model, classifier)
        
        return combined_model, tokenizer

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