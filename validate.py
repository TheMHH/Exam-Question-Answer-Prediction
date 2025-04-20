import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model_loader import ModelLoader
from data_loader import ExamDataLoader
from constants import IDX_TO_ANSWER

class ExamEvaluator:
    def __init__(
        self,
        model_type: str,
        batch_size: int = 8,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_type (str): Type of model to use ("gemma-it" or "gemma-pt")
            batch_size (int): Batch size for evaluation
            checkpoint_path (Optional[str]): Path to a checkpoint to evaluate
            device (str): Device to evaluate on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        
        # Initialize model and data loaders
        self.model_loader = ModelLoader(model_type, device)
        self.data_loader = ExamDataLoader(batch_size=batch_size)
        self.model, self.tokenizer = self.model_loader.load_from_checkpoint(checkpoint_path)
        
        # Answer mapping for converting numeric predictions back to letters
        self.idx_to_answer = IDX_TO_ANSWER

    def evaluate(self) -> Dict:
        """
        Evaluate the model on the test dataset.
        
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        # Get test dataloader
        _, test_loader = self.data_loader.get_all_splits(self.tokenizer)
        
        # Initialize lists for predictions and true labels
        all_predictions = []
        all_labels = []
        
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
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    output_hidden_states=True
                )
                
                # Get probabilities from the combined model
                probabilities = outputs.probabilities
                
                # Calculate loss
                loss = F.cross_entropy(probabilities, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )
        
        # Convert some predictions to answer letters for display
        pred_answers = [self.idx_to_answer[pred] for pred in all_predictions[:5]]
        true_answers = [self.idx_to_answer[label] for label in all_labels[:5]]
        
        # Prepare results
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nSample Predictions (first 5):")
        print(f"Predicted: {pred_answers}")
        print(f"Actual: {true_answers}")
        
        return results 