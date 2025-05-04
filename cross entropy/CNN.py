import os
import pandas as pd
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights, efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

####################################

from tqdm.auto import tqdm  # For progress bars
import wandb
import json
import pprint

################################################################################
# Define a custom dataset class
################################################################################
class CustomDataset(Dataset):
    def __init__(self, df, is_train=True, transform=None):
        self.df = df
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['file_name']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            label = self.df.iloc[idx]['label']
            return image, label
        else:
            return image, -1

################################################################################
# Define a custom model
################################################################################

class CNNBasedClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_backbone=True):
        super(CNNBasedClassifier, self).__init__()

        if model_name == "ResNet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "ResNeXt50":
            model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        elif model_name == "EfficientNetB3":
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        
        self.model = model

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Classifier head
        if model_name == "EfficientNetB3":
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        return self.model(x)
    


################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()   ### TODO
        _, predicted = outputs.max(1)    ### TODO

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs, labels)    ### TODO -- loss calculation

            running_loss += loss.item()  ### SOLUTION -- add loss from this sample
            _, predicted = outputs.max(1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    #    Configuration Dictionary
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "EfficientNetB3",   # Change name when using a different model
        "batch_size": 16, # run batch size finder to find optimal batch size
        "image_size": 300, # Resize images to this size
        "learning_rate": 3e-4,
        "epochs": 10,  
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "ai-human-image-classification",
        "seed": 42,
    }

    best_model_filename = f"best_model_{CONFIG['model']}.pth"

    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation
    ############################################################################

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),                
        transforms.RandomResizedCrop(CONFIG['image_size']),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),     
        transforms.RandomRotation(20),           
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]) 

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),            
        transforms.ToTensor(),                        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    base_dir = '../dataset/'

    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))


    test_df = test_df.rename(columns={'id': 'file_name'})
    train_df = train_df.drop(columns=['Unnamed: 0'])

    def modify_image_path(df):
        # Modify the file_name to have the full path
        df['file_name'] = df['file_name'].apply(lambda x: os.path.join(base_dir, x))
        return df

    train_df = modify_image_path(train_df)
    test_df = modify_image_path(test_df)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    ### Define the train and validation datasets, dataloaders
    trainset = CustomDataset(train_df, is_train=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    valset = CustomDataset(val_df, is_train=True, transform=transform_test)
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    testset = CustomDataset(test_df, is_train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################

    # Instantiate the model.
    model = CNNBasedClassifier(model_name=CONFIG['model'], num_classes=2, freeze_backbone=True) # instantiate your model ### TODO

    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    
    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    # Label Smoothing for generalization.
    criterion = torch.nn.CrossEntropyLoss()

    # Weight decay is added to the optimizer for regularization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)

    # CosineAnnealingLR is used to reduce the learning rate as the training progresses for smoother convergence.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Initialize wandb
    wandb.init(project="ai-human-image-classification", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_filename)
            wandb.save(best_model_filename) # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################

    model.load_state_dict(torch.load(best_model_filename))
    model.eval()
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:  # Use val_loader or test_loader
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += preds.eq(labels.to(CONFIG["device"])).sum().item()

    # Compute metrics
    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    predictions = []

    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(CONFIG["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # --- Create Submission File ---
    submission_df = pd.DataFrame({'id': test_df.iloc[:, 0], 'label': predictions})
    submission_df['id'] = submission_df['id'].str.replace('./dataset/', '', regex=False)
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv created successfully.")



if __name__ == '__main__':
    main()