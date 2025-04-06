import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm  # Using timm for model instantiation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm  # For progress bars
import wandb
import pprint

best_model_filename = "best_transformer_model.pth"

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
            return image

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
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
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    # Compute F1 Score
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    return val_loss, val_acc, val_f1, all_preds, all_labels

################################################################################
# Main function for setting up the Vision Transformer experiment with wandb
################################################################################
def main(test_only=False):
    CONFIG = {
        "model": "ViT-B/32",  # Using a base Vision Transformer model
        "batch_size": 16,      
        "learning_rate": 3e-4,
        "epochs": 8,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "ai-human-image-classification-transformer",
        "seed": 42,
    }

    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    # Data Transformations with more aggressive augmentations
    ############################################################################
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # up to 15 degrees rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.33))  # aggressive random erasing
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    ############################################################################
    # Data Loading
    ############################################################################
    base_dir = '../dataset/'
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))

    test_df = test_df.rename(columns={'id': 'file_name'})
    train_df = train_df.drop(columns=['Unnamed: 0'])

    def modify_image_path(df):
        df['file_name'] = df['file_name'].apply(lambda x: os.path.join(base_dir, x))
        return df

    train_df = modify_image_path(train_df)
    test_df = modify_image_path(test_df)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    trainset = CustomDataset(train_df, is_train=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    valset = CustomDataset(val_df, is_train=True, transform=transform_val)
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    ############################################################################
    # Instantiate the Vision Transformer model without stochastic depth
    ############################################################################
    model = timm.create_model('vit_base_patch32_224', pretrained=True)
    # Replace the classifier head with an enhanced classifier
    in_features = model.head.in_features
    model.head = torch.nn.Sequential(
        torch.nn.Linear(in_features, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 2)
    )
    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(model)

    ############################################################################
    # Loss, Optimizer, and Scheduler
    ############################################################################
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    if not test_only:
        print("Training starts now.")
        ############################################################################
        # Initialize wandb for experiment tracking
        ############################################################################
        wandb.login()
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
        wandb.watch(model)

        ############################################################################
        # Training Loop
        ############################################################################
        best_val_acc = 0.0
        best_val_f1 = 0.0
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
            val_loss, val_acc, val_f1, all_preds, all_labels = validate(model, valloader, criterion, CONFIG["device"])
            scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"]
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_filename)
                wandb.save(best_model_filename)

        wandb.finish()

        ############################################################################
        # Evaluation on Validation Set (for logging final metrics)
        ############################################################################
        model.load_state_dict(torch.load(best_model_filename))
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(valloader, desc="Evaluating Validation Set"):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        final_f1 = f1_score(all_labels, all_preds, average='binary')
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {final_f1:.4f}')
        
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG)  # Reinitialize to log final metric if desired
        wandb.finish()
    else:
        # If test_only, load the best model directly.
        model.load_state_dict(torch.load(best_model_filename))
        model.eval()

    ############################################################################
    # Test Prediction and Submission Creation for Kaggle
    ############################################################################
    print("Testing starts now.")
    testset = CustomDataset(test_df, is_train=False, transform=transform_val)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(testloader, desc="Predicting Test Set"):
            images = images.to(CONFIG["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission_df = pd.DataFrame({
        'id': test_df['file_name'].apply(lambda x: x.replace('../dataset/', '')),
        'label': predictions
    })
    submission_dir = "../results/ViT-B32/advanced"
    os.makedirs(submission_dir, exist_ok=True)
    submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

    print("submission.csv created successfully.")

if __name__ == '__main__':
    main(test_only=False)
