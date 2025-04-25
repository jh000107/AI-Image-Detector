import os
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from util import ContrastiveImageDataset
from networks.efficientnet import SupConEfficientNet, LinearClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm.auto import tqdm  # For progress bars
import wandb


def modify_image_path(df):
    # Modify the file_name to have the full path
    df['file_name'] = df['file_name'].apply(lambda x: os.path.join('./dataset/', x))
    return df


def set_loader(CONFIG):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=CONFIG['image_size'], scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),     
        transforms.RandomRotation(20),           
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    train_df = pd.read_csv(CONFIG["train_csv_path"])
    test_df = pd.read_csv(CONFIG["test_csv_path"])


    test_df = test_df.rename(columns={'id': 'file_name'})
    train_df = train_df.drop(columns=['Unnamed: 0'])

    train_df = modify_image_path(train_df)
    test_df = modify_image_path(test_df)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    train_dataset = ContrastiveImageDataset(df=train_df, is_train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

    val_dataset = ContrastiveImageDataset(df=val_df, is_train=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    test_dataset = ContrastiveImageDataset(df=test_df, is_train=False, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    return train_loader, val_loader, test_loader, test_df


def train(epoch, encoder, classifier, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch for supervised contrastive learning (SupCon)."""
    device = CONFIG["device"]
    encoder.eval()
    classifier.train()
    total, correct, total_loss = 0, 0, 0.0


    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [SupConClassifier]", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            features = encoder(images)
            features = torch.flatten(features, 1)

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    
    return total_loss / len(trainloader), acc

def validate(epoch, encoder, classifier, valloader, criterion, CONFIG):
    """Validate the model on the validation set."""
    device = CONFIG["device"]
    encoder.eval()
    classifier.eval()
    total, correct, total_loss = 0, 0, 0.0

    progress_bar = tqdm(valloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Validation]", leave=False)

    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            features = encoder(images)
            features = torch.flatten(features, 1)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return total_loss / len(valloader), acc


def main():

    CONFIG = {
        "model": "SupConEfficientNetB3-linear",   # Change name when using a different model
        "ckpt_path": "./supcon_efficientnet.pth", # Path to the pretrained model"
        "batch_size": 64, # run batch size finder to find optimal batch size
        "image_size": 300, # Resize images to this size
        "learning_rate": 0.003,
        "epochs": 10,  
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "SupCon-classifier-ai-image-detection",
        "train_csv_path": "./dataset/train.csv",
        "test_csv_path": "./dataset/test.csv",
        "seed": 42,
    }

    # build data loader
    train_loader, val_loader, test_loader, test_df = set_loader(CONFIG)

    # define model, criterion, and optimizer
    model = SupConEfficientNet(pretrained=True, head='mlp', feat_dim=128)
    ckpt = torch.load(CONFIG['ckpt_path'], map_location=CONFIG['device'])
    model.load_state_dict(ckpt)
    encoder = model.encoder.to(CONFIG['device'])

    classifier = LinearClassifier(num_classes=2).to(CONFIG['device'])

    criterion = nn.CrossEntropyLoss()
    # Weight decay is added to the optimizer for regularization.
    optimizer = torch.optim.SGD(classifier.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=1e-4)

    # CosineAnnealingLR is used to reduce the learning rate as the training progresses for smoother convergence.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    print("\nModel summary:")
    print(f"{model}\n")

    print("\nEncoder summary:")
    print(f"{encoder}\n")


    # Initialize wandb
    wandb.init(project=CONFIG['wandb_project'], config=CONFIG)
    wandb.watch(classifier)  # watch the model gradients

    ############################################################################
    # --- Training Loop ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, encoder, classifier, train_loader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(epoch, encoder, classifier, val_loader, criterion, CONFIG)
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
            torch.save(classifier.state_dict(), "best_supcon_efficientnet_classifier.pth")
            wandb.save("best_supcon_efficientnet_classifier.pth") # Save to wandb as well


    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################

    classifier.load_state_dict(torch.load("best_supcon_efficientnet_classifier.pth"))
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:  # Use val_loader or test_loader
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            features = encoder(images)
            features = torch.flatten(features, 1)
            outputs = classifier(features)
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
        for images, _ in test_loader:
            images = images.to(CONFIG["device"])
            features = encoder(images)
            features = torch.flatten(features, 1)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # --- Create Submission File ---
    submission_df = pd.DataFrame({'id': test_df.iloc[:, 0], 'label': predictions})
    submission_df['id'] = submission_df['id'].str.replace('./dataset/', '', regex=False)
    submission_df.to_csv("submission2.csv", index=False)
    print("submission2.csv created successfully.")

if __name__ == '__main__':
    main()