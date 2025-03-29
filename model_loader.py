from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
import json
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
        
        self.model_configs = {
            "gemma-it": {
                "name": "google/gemma-3-4b-it",
                "max_length": 8192,
                "hidden_size": 3072
            },
            "gemma-pt": {
                "name": "google/gemma-3-4b-pt",
                "max_length": 8192,
                "hidden_size": 3072
            }
        }
        
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.config = self.model_configs[model_type]
        
        # Simplified classifier - single linear layer
        self.classifier = nn.Linear(self.config["hidden_size"], len(IDX_TO_ANSWER))

    def _create_combined_model(self, base_model: nn.Module) -> nn.Module:
        """
        Create a combined model with the base model and classifier.
        
        Args:
            base_model (nn.Module): The base model to combine with classifier
            
        Returns:
            nn.Module: Combined model
        """
        model_dtype = next(base_model.parameters()).dtype

        classifier = self.classifier.to(self.device, dtype=model_dtype)
        
        # Initialize classifier weights
        nn.init.xavier_normal_(classifier.weight)
        nn.init.zeros_(classifier.bias)
        
        class CombinedModel(nn.Module):
            def __init__(self, base_model, classifier):
                super().__init__()
                self.base_model = base_model
                self.classifier = classifier
                self.is_combined_model = True
            
            def forward(self, **inputs):
                # Get base model outputs
                outputs = self.base_model(**inputs, output_hidden_states=True)
                
                # Extract last hidden state from the last layer
                hidden_states = outputs.hidden_states[-1]  # Always use the last layer
                last_hidden_state = hidden_states[:, -1, :]
                
                # Apply classifier
                logits = self.classifier(last_hidden_state)
                
                # Apply temperature scaling
                temperature = 0.5
                scaled_logits = logits / temperature
                
                # Get probabilities
                probabilities = F.softmax(scaled_logits, dim=1)
                
                # Add to outputs
                outputs.logits = logits
                outputs.probabilities = probabilities
                
                return outputs
            
            def save_pretrained(self, save_dir: str, **kwargs):
                """
                Save both the base model and classifier state
                """
                self.base_model.save_pretrained(save_dir, **kwargs)
                
                classifier_path = os.path.join(save_dir, "classifier.pt")
                torch.save(self.classifier.state_dict(), classifier_path)
                
                # Save config
                model_config = {
                    "is_combined_model": True,
                    "classifier_path": "classifier.pt"
                }
                with open(os.path.join(save_dir, "combined_model_config.json"), 'w') as f:
                    json.dump(model_config, f)
        
        return CombinedModel(base_model, classifier)

    def load_for_finetuning(self) -> Tuple[nn.Module, nn.Module]:
        """
        Load the model with an added classifier layer for fine-tuning.
        
        Returns:
            Tuple[nn.Module, nn.Module]: Combined model and processor
        """
        processor = AutoProcessor.from_pretrained(self.config["name"])
        
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.config["name"],
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True,
            trust_remote_code=True
        ).to(self.device)
        
        combined_model = self._create_combined_model(base_model)
        
        return combined_model, processor

    def load_from_checkpoint(self, checkpoint_path: str) -> Tuple[nn.Module, nn.Module]:
        """
        Load a fine-tuned model from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the fine-tuned model checkpoint
            
        Returns:
            Tuple[nn.Module, nn.Module]: Fine-tuned model and processor
        """
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        base_model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True,
            trust_remote_code=True
        ).to(self.device)
        
        combined_config_path = os.path.join(checkpoint_path, "combined_model_config.json")
        
        if os.path.exists(combined_config_path):
            with open(combined_config_path, 'r') as f:
                combined_config = json.load(f)
            
            if combined_config.get("is_combined_model", False):
                combined_model = self._create_combined_model(base_model)
                
                classifier_path = os.path.join(checkpoint_path, combined_config["classifier_path"])
                if os.path.exists(classifier_path):
                    classifier_state_dict = torch.load(classifier_path, map_location=self.device)
                    combined_model.classifier.load_state_dict(classifier_state_dict)
                
                return combined_model, processor
        
        return base_model, processor

    def load_base_model(self) -> Tuple[nn.Module, nn.Module]:
        """
        Load just the base model for text generation, without the classifier.
        
        Returns:
            Tuple[nn.Module, nn.Module]: Base model and processor
        """
        processor = AutoProcessor.from_pretrained(self.config["name"])
        
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.config["name"],
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        return base_model, processor

    def save_model(self, model: nn.Module, processor: nn.Module, save_path: str) -> None:
        """
        Save the current model and processor.
        
        Args:
            model (nn.Module): Model to save
            processor (nn.Module): Processor to save
            save_path (str): Path to save the model and processor
        """
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
