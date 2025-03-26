import torch
from tqdm import tqdm
from typing import Dict, List
from evaluate import load
from model_loader import ModelLoader
from data_loader import DataLoader

class Evaluator:
    def __init__(
        self,
        model_type: str,
        batch_size: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_type (str): Type of model to use ("gemma-it" or "gemma-pt")
            batch_size (int): Batch size for evaluation
            device (str): Device to evaluate on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        
        # Initialize model and data loaders
        self.model_loader = ModelLoader(model_type, device)
        self.data_loader = DataLoader(batch_size=batch_size)
        self.model, self.tokenizer = self.model_loader.load_model()
        
        # Initialize metrics
        self.bleu = load("bleu")
        self.rouge = load("rouge")

    def evaluate(self, checkpoint_path: str = None) -> Dict:
        """
        Evaluate the model on the test dataset.
        
        Args:
            checkpoint_path (str): Path to a checkpoint to evaluate
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        if checkpoint_path:
            self.model, self.tokenizer = self.model_loader.load_model(checkpoint_path)
        
        # Get test dataloader
        _, _, test_loader = self.data_loader.get_all_splits(self.tokenizer)
        
        # Initialize lists for predictions and references
        predictions: List[str] = []
        references: List[str] = []
        
        # Evaluation phase
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode predictions and references
                batch_predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                batch_references = self.tokenizer.batch_decode(
                    input_ids,
                    skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        
        # Calculate BLEU score
        bleu_score = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references
        )
        
        # Prepare results
        results = {
            'loss': avg_loss,
            'bleu': bleu_score['bleu'],
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        }
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"BLEU Score: {bleu_score['bleu']:.4f}")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return results 