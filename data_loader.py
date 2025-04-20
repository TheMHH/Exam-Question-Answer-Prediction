from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import os
from constants import EXAM_QUESTION_CHAT_TEMPLATE, ANSWER_TO_IDX

class ImageTextDataset(Dataset):
    def __init__(self, dataset, processor, max_length = 512):
        """
        Initialize the dataset.
        
        Args:
            dataset (Dict): Dataset dictionary from Hugging Face datasets
            processor: Processor for text and image processing
            max_length (int): Maximum sequence length for tokenization
        """
        self.dataset = dataset.filter(ImageTextDataset.is_valid_example)
        self.processor = processor
        self.max_length = max_length
        self.answer_mapping = ANSWER_TO_IDX
    
    @staticmethod
    def is_valid_example(example):
        key = example['answer_key']
        return key in ANSWER_TO_IDX or key.upper() in ANSWER_TO_IDX

    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing processed image and label
        """
        item = self.dataset[idx]
        
        image = item['image']
        
        label = self.answer_mapping[item['answer_key']]
        
        prompt = self.processor.apply_chat_template(
            EXAM_QUESTION_CHAT_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ExamDataLoader:
    def __init__(self, batch_size: int = 8, max_length: int = 512):
        """
        Initialize the data loader.
        
        Args:
            batch_size (int): Batch size for data loading
            max_length (int): Maximum sequence length for tokenization
        """
        self.batch_size = batch_size
        self.max_length = max_length

    def get_dataloader(self, processor, shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            processor: Processor for text and image processing
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        dataset = ImageTextDataset(
            load_dataset("Rocktim/EXAMS-V"),
            processor,
            max_length=self.max_length
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def get_all_splits(self, processor) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get DataLoaders for all dataset splits.
        
        Args:
            processor: Processor for text and image processing
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders
        """
        # Load all splits
        train_dataset = load_dataset("Rocktim/EXAMS-V", split="train")
        test_dataset = load_dataset("Rocktim/EXAMS-V", split="test")
        
        # Create datasets
        train_ds = ImageTextDataset(train_dataset, processor, self.max_length)
        test_ds = ImageTextDataset(test_dataset, processor, self.max_length)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
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
        
        return train_loader, test_loader 