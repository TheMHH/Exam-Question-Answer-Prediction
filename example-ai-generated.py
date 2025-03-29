import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# Define a custom dataset class for image-text data
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, processor, label2id=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.label2id = label2id or {label: i for i, label in enumerate(set(labels))}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension added by the processor
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Convert text label to ID
        label_id = self.label2id[label]
        
        return {
            **inputs,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

# Configuration
model_name = "Salesforce/blip-image-captioning-base"  # Example image-text model
batch_size = 8
num_epochs = 5
learning_rate = 5e-5
warmup_steps = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Function to load sample data
def load_sample_data():
    # Replace with your actual data loading code
    # This should return:
    # - A list of paths to your images
    # - A list of class labels as strings
    
    # Example (replace with actual data):
    image_paths = [
        "./data/images/cat_1.jpg",
        "./data/images/dog_1.jpg",
        "./data/images/cat_2.jpg",
    ]
    
    labels = ["cat", "dog", "cat"]
    
    return image_paths, labels

# Load data
image_paths, labels = load_sample_data()

# Get unique labels
unique_labels = sorted(set(labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

# Replace the language modeling head with a classification head
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

# Modify the model to use a classification head
# Save the original LM head for later reference
original_lm_head = model.lm_head

# Replace with classification head
model.lm_head = ClassificationHead(model.config.hidden_size, num_labels)
model.config.id2label = id2label
model.config.label2id = label2id

# Move model to device
model = model.to(device)

# Split data
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = ImageClassificationDataset(train_image_paths, train_labels, processor, label2id)
val_dataset = ImageClassificationDataset(val_image_paths, val_labels, processor, label2id)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

# Custom training function for classification
def train():
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        # Move batch to device
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass - get the last hidden state
        outputs = model(pixel_values=pixel_values, return_dict=True)
        
        # Get hidden state of the first token in the sequence
        hidden_states = outputs.decoder_hidden_states[-1][:, 0, :]
        
        # Pass through classification head
        logits = model.lm_head(hidden_states)
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate():
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, return_dict=True)
            
            # Get hidden state of the first token
            hidden_states = outputs.decoder_hidden_states[-1][:, 0, :]
            
            # Pass through classification head
            logits = model.lm_head(hidden_states)
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return total_loss / len(val_loader), accuracy

# Training process
print(f"Training on {device}")
print(f"Number of classes: {num_labels}")
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss = train()
    print(f"Training loss: {train_loss:.4f}")
    
    # Evaluate
    val_loss, val_accuracy = evaluate()
    print(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Saving best model...")
        
        # Create directory if it doesn't exist
        os.makedirs("./fine_tuned_image_classifier", exist_ok=True)
        
        # Save the model with classification head
        model.save_pretrained("./fine_tuned_image_classifier")
        processor.save_pretrained("./fine_tuned_image_classifier")

print("Training complete!")

# Function to load and use the fine-tuned model
def load_and_predict(image_path):
    # Load model and processor
    loaded_processor = AutoProcessor.from_pretrained("./fine_tuned_image_classifier")
    loaded_model = AutoModelForVision2Seq.from_pretrained("./fine_tuned_image_classifier")
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    inputs = loaded_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = loaded_model(pixel_values=pixel_values, return_dict=True)
        hidden_states = outputs.decoder_hidden_states[-1][:, 0, :]
        logits = loaded_model.lm_head(hidden_states)
        pred_idx = torch.argmax(logits, dim=1).item()
        predicted_label = loaded_model.config.id2label[pred_idx]
    
    return predicted_label

# Example usage
# predicted_class = load_and_predict("path/to/new/image.jpg")
# print(f"Predicted class: {predicted_class}")