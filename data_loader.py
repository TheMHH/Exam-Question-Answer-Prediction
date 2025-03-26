from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import os

class ImageTextDataset(Dataset):
    def __init__(self, dataset: Dict, tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            dataset (Dict): Dataset dictionary from Hugging Face datasets
            tokenizer: Tokenizer for text processing
            max_length (int): Maximum sequence length for tokenization
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing processed image and text data
        """
        item = self.dataset[idx]
        
        # Process image
        image = Image.open(item['image']).convert('RGB')
        
        # Process text
        text = item['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class DataLoader:
    def __init__(self, batch_size: int = 8, max_length: int = 512):
        """
        Initialize the data loader.
        
        Args:
            batch_size (int): Batch size for data loading
            max_length (int): Maximum sequence length for tokenization
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset = None

    def load_dataset(self, split: str = "train") -> None:
        """
        Load the Rocktim/EXAMS-V dataset.
        
        Args:
            split (str): Dataset split to load ("train", "validation", or "test")
        """
        self.dataset = load_dataset("Rocktim/EXAMS-V", split=split)

    def get_dataloader(self, tokenizer, shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            tokenizer: Tokenizer for text processing
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        if self.dataset is None:
            raise ValueError("Dataset must be loaded before creating DataLoader")
        
        dataset = ImageTextDataset(
            self.dataset,
            tokenizer,
            max_length=self.max_length
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def get_all_splits(self, tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get DataLoaders for all dataset splits.
        
        Args:
            tokenizer: Tokenizer for text processing
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders
        """
        # Load all splits
        train_dataset = load_dataset("Rocktim/EXAMS-V", split="train")
        val_dataset = load_dataset("Rocktim/EXAMS-V", split="validation")
        test_dataset = load_dataset("Rocktim/EXAMS-V", split="test")
        
        # Create datasets
        train_ds = ImageTextDataset(train_dataset, tokenizer, self.max_length)
        val_ds = ImageTextDataset(val_dataset, tokenizer, self.max_length)
        test_ds = ImageTextDataset(test_dataset, tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader 