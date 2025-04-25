import os
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from util import TwoCropTransform, ContrastiveImageDataset

from networks.resnet50 import SupConResNet
from networks.efficientnet import SupConEfficientNet

from losses import SupConLoss


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
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    train_dataset = ContrastiveImageDataset(
        df = modify_image_path(pd.read_csv(CONFIG['train_csv_path'])),
        is_train=True,
        transform=TwoCropTransform(train_transform)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

    return train_loader


def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch for supervised contrastive learning (SupCon)."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [SupCon]", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        # images is a tuple of (view1, view2)
        x1, x2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)
        bsz = labels.size(0)

        # create the combined batch: [2B, C, H, W]
        combined = torch.cat([x1, x2], dim=0)
        features = model(combined)  # [2B, D]

        # split into two views and recombine: [B, 2, D]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # compute SupCon loss
        loss = criterion(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})

    train_loss = running_loss / len(trainloader)
    
    return train_loss


def main():

    CONFIG = {
        "model": "SupConResNet50",   # Change name when using a different model
        "batch_size": 64, # run batch size finder to find optimal batch size
        "image_size": 224, # Resize images to this size
        "learning_rate": 0.01,
        "temperature": 0.07,
        "epochs": 10,  
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "SupCon-ai-image-detection",
        "train_csv_path": "./dataset/train.csv",
        "seed": 42,
    }

    # build data loader
    train_loader = set_loader(CONFIG)

    # define model, criterion, and optimizer
    model = SupConResNet(pretrained=True, head='mlp', feat_dim=128)
    model.to(CONFIG['device'])

    criterion = SupConLoss(temperature=CONFIG['temperature'])
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])


    # Initialize wandb
    wandb.init(project=CONFIG['wandb_project'], config=CONFIG)
    wandb.watch(model)  # watch the model gradients


    # training routine
    for epoch in range(CONFIG['epochs']):


        # train for one epoch
        loss = train(epoch, model, train_loader, optimizer, criterion, CONFIG)
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch+1,
            "train_loss": loss,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

    # save the last model
    torch.save(model.state_dict(), "supcon_resnet50_batch_128.pth")
    wandb.save("supcon_resnet50_batch_128.pth") # Save to wandb as well

    wandb.finish()

if __name__ == '__main__':
    main()