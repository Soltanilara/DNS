import os
import pandas as pd
import wandb
import torch
import torch.nn as nn
from SteerModel import EfficientNet, TRAIN_TRANSFORMATIONS
from SteerDataset import SteerDataset
import yaml
import argparse

# Define a list of required parameters
REQUIRED_PARAMS = [
    "name", "project", "epochs", "batch_size", "lr", "architecture",
    "seed", "device", "dataset_dir", "train_annot_csv", "model_save_dir",
    "resume_checkpoint", "start_epoch", 
]

def check_required_params(config):
    for param in REQUIRED_PARAMS:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' is missing from the configuration.")

# Function to train the model
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        direction = batch["direction"].to(device)
        steering_angle = torch.reshape(batch["steering_angle"], (-1, 1)).to(device)

        # Compute prediction error
        pred = model(image, direction)

        # Combine steering_angle for y
        y = steering_angle
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(image)
            train_loss += loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(f"Train Error: Avg loss: {train_loss:>8f} \n")
    return train_loss

# Function to evaluate the model
def eval(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            direction = batch["direction"].to(device)
            steering_angle = torch.reshape(batch["steering_angle"], (-1, 1)).to(device)

            # Compute prediction error
            pred = model(image, direction)

            # Combine steering_angle and throttle for y
            y = steering_angle
            eval_loss += loss_fn(pred, y).item()

    eval_loss /= num_batches
    print(f"Evaluation Error: Avg loss: {eval_loss:>8f} \n")
    return eval_loss

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Train a steering model")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config_file, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Check if all required parameters are present
    check_required_params(config)

    # Initialize WandB for experiment tracking
    wandb.init(
        name=config["name"],
        project=config["project"],
        config=config,
        save_code=True,
    )
    config = wandb.config

    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Load training data
    train_df = pd.read_csv(config.train_annot_csv)
    train_df['path'] = train_df['path'].apply(lambda path: os.path.join(config.dataset_dir, path))
    train_dataset = SteerDataset(train_df, transform=TRAIN_TRANSFORMATIONS, flip=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    # Initialize model
    device = config.device
    model = EfficientNet().to(device)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    # Check if resuming from a checkpoint
    if config.resume_checkpoint:
        checkpoint = torch.load(config.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if config.start_epoch is not None:
            start_epoch = config.start_epoch
        else:
            start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)

        # Log training metrics to WandB
        metrics = {
            "train/train_loss": train_loss,
            "train/epoch": epoch+1,
        }
        wandb.log(metrics)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.model_save_dir, f"{wandb.run.project}_{wandb.run.name}_{config.architecture}_epoch-{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            wandb.save(checkpoint_path)

    # Finish the WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
