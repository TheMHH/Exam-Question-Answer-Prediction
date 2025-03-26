import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from typing import Dict, Optional
from transformers import get_scheduler
from model_loader import ModelLoader
from data_loader import DataLoader

class Trainer:
    def __init__(
        self,
        model_type: str,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type (str): Type of model to use ("gemma-it" or "gemma-pt")
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimization
            num_epochs (int): Number of training epochs
            checkpoint_dir (str): Directory to save checkpoints
            device (str): Device to train on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize model and data loaders
        self.model_loader = ModelLoader(model_type, device)
        self.data_loader = DataLoader(batch_size=batch_size)
        self.model, self.tokenizer = self.model_loader.load_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs
        )

    def train(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Train the model.
        
        Args:
            checkpoint_path (Optional[str]): Path to a checkpoint to resume training from
        """
        if checkpoint_path:
            self.model, self.tokenizer = self.model_loader.load_model(checkpoint_path)
        
        # Get data loaders
        train_loader, val_loader, _ = self.data_loader.get_all_splits(self.tokenizer)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
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
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    images = batch['image'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images
                    )
                    
                    total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.model_type}_epoch_{epoch + 1}.pt"
                )
                self.model_loader.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save final checkpoint
            if epoch == self.num_epochs - 1:
                final_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.model_type}_final.pt"
                )
                self.model_loader.save_model(final_checkpoint_path)
                print(f"Saved final checkpoint to {final_checkpoint_path}") 