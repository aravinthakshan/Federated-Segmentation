from model.unet import UNet
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from dataloader import BraTSDataset
from misc import hausdorff_distance_95, dice_coefficient, get_train_transforms, get_val_transforms

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{epoch_loss / (pbar.n + 1):.4f}")
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    dice_scores = {'WT': [], 'TC': [], 'ET': []}
    hd95_scores = {'WT': [], 'TC': [], 'ET': []}
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for batch in pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                epoch_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                
                for i, region in enumerate(['WT', 'TC', 'ET']):
                    pred_numpy = preds[:, i].cpu().numpy()
                    mask_numpy = masks[:, i].cpu().numpy()
                    
                    # Calculate Dice scores for non-empty masks
                    for j in range(len(pred_numpy)):
                        if np.sum(mask_numpy[j]) > 0:  # Only calculate if ground truth has tumor
                            try:
                                dice = dice_coefficient(mask_numpy[j], pred_numpy[j])
                                dice_scores[region].append(dice)
                                
                                # Calculate HD95 for non-empty masks
                                if np.sum(pred_numpy[j]) > 0:  # Only if prediction has tumor
                                    hd = hausdorff_distance_95(mask_numpy[j], pred_numpy[j])
                                    hd95_scores[region].append(hd)
                            except Exception as e:
                                print(f"Error calculating metrics for {region}: {e}")
                
                current_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=f"{current_loss:.4f}")
    

    mean_dice = {k: np.mean(v) if v else 0.0 for k, v in dice_scores.items()}
    mean_hd95 = {k: np.mean(v) if v else float('inf') for k, v in hd95_scores.items()}
    
    return epoch_loss / len(dataloader), mean_dice, mean_hd95

def main():

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 3e-4
    img_size = 240
    
    # Dataset path
    brats_root = "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training" 
    
    # Prepare patient list
    all_patients = []
    for grade_dir in ["HGG", "LGG"]:
        grade_path = os.path.join(brats_root, grade_dir)
        if os.path.exists(grade_path):
            for patient_dir in sorted(os.listdir(grade_path)):
                patient_path = os.path.join(grade_path, patient_dir)
                if os.path.isdir(patient_path):
                    all_patients.append({
                        'id': patient_dir,
                        'path': patient_path,
                        'grade': 'HGG' if grade_dir == 'HGG' else 'LGG'
                    })
    
    print(f"Total patients found: {len(all_patients)}")
    
    #  train and validation sets (80/20)
    train_patients, val_patients = train_test_split(
        all_patients, test_size=0.2, random_state=seed
    )
    
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    
    # Create datasets
    train_dataset = BraTSDataset(
        train_patients,
        transform=get_train_transforms(size=img_size),
        mode='train',
        slice_selection_method='tumor_only'
    )
    
    val_dataset = BraTSDataset(
        val_patients,
        transform=get_val_transforms(size=img_size),
        mode='val',
        slice_selection_method='tumor_only'
    )
    
    print(f"Training slices: {len(train_dataset)}")
    print(f"Validation slices: {len(val_dataset)}")
    
    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model, loss, and optimizer
    model = UNet(in_channels=4, out_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create directory for saving model and results
    os.makedirs("results", exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_dice_wt = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_dice, val_hd95 = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("Mean Dice (DSC):")
        for k, v in val_dice.items():
            print(f"  {k}: {v:.4f}")
        print("Mean HD95:")
        for k, v in val_hd95.items():
            print(f"  {k}: {v:.4f}")
        
        # Save best model
        if val_dice['WT'] > best_dice_wt:
            best_dice_wt = val_dice['WT']
            print(f"Saving new best model with WT Dice: {best_dice_wt:.4f}")
            torch.save(model.state_dict(), os.path.join("results", "best_model.pth"))
        
        # Also save if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving new best model with Val Loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join("results", "best_model_loss.pth"))
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_hd95': val_hd95,
        }, os.path.join("results", "last_checkpoint.pth"))

if __name__ == "__main__":
    main()
