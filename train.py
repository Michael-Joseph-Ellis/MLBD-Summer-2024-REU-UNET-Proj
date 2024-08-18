# Necessary imports
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms as transforms
from pyimagesearch.dataset2 import SegmentationDataset
from pyimagesearch.model3 import UNet
from pyimagesearch import config

class_names = ["Background", "Sedan", "Pickup", "Off-road"]

# Define the SegmentationDataset class
class SegmentationDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_paths = []
        self.mask_paths = []

        for trail_name in os.listdir(self.root):
            split_path = os.path.join(root, trail_name, split.capitalize())
            imgs_path = os.path.join(split_path, "imgs")
            masks_path = os.path.join(split_path, "annos/int_maps/")
            for mask_name in os.listdir(masks_path):
                mask_path = os.path.join(masks_path, mask_name)
                self.mask_paths.append(mask_path)
                image_name = "img_" + mask_name.split("_", 1)[1]
                image_path = os.path.join(imgs_path, image_name)
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return (image, mask)

# Define a Compose class to handle transformations
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        target = np.array(target)
        target = torch.tensor(target, dtype=torch.int64)
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        return image, target

# Define transformations for the dataset
transform = []
transform.append(transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST))
transform = Compose(transform)

# Initialize the training and testing datasets
trainDS = SegmentationDataset('/home/mje2/projects/REU 2024/Transformers Stuff/CaT/CAT', split='Train', transforms=transform)
testDS = SegmentationDataset('/home/mje2/projects/REU 2024/Transformers Stuff/CaT/CAT', split='Test', transforms=transform)

# Initialize the UNet model
unet = UNet(in_channels=3, classes=4).to(config.DEVICE)

# Define the loss function and optimizer
lossFunc = CrossEntropyLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# Calculate steps per epoch for training and test sets
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# Function to calculate mean Intersection over Union (mIoU)
def calc_miou(model, val_loader, num_classes=4):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            predicted = predicted.to('cpu')
            labels = labels.to('cpu')
            predictions.append(predicted.numpy())
            targets.append(labels.numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0).astype(np.int64)
    targets = targets.flatten()
    predictions = predictions.flatten()
    confusion_mat = confusion_matrix(targets, predictions)
    iou_per_class = np.diag(confusion_mat) / (confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat))
    mean_iou = np.nanmean(iou_per_class)
    print(f"[DEBUG] Mean IoU: {mean_iou}")
    for class_idx, class_name in enumerate(class_names):
        print(f"[DEBUG] IoU for {class_name}: {iou_per_class[class_idx]}")
    return mean_iou, iou_per_class, confusion_mat

# Define the output directory and model path
output_dir = '/home/mje2/projects/REU 2024/Transformers Stuff/Work with CAVs Dataset (CaT)/unetMLBD/output'
config.MODEL_PATH = os.path.join(output_dir, 'unet_model.pth')

# Function to train the model and calculate mIoU
def train_model_and_calc_miou(unet, trainLoader, testLoader, num_epochs, config, output_dir):
    train_losses = []
    miou_per_epoch = []
    iou_per_class_history = []
    lossFunc = CrossEntropyLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)
    trainSteps = len(trainLoader)
    testSteps = len(testLoader)

    print("[INFO] training the network...")
    startTime = time.time()
    for e in range(num_epochs):
        unet.train()
        totalTrainLoss = 0
        trainLoaderIter = tqdm(trainLoader)
        for (i, (x, y)) in enumerate(trainLoaderIter):
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = unet(x)
            loss = lossFunc(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss.item()
        avgTrainLoss = totalTrainLoss / trainSteps
        train_losses.append(avgTrainLoss)

        print(f"[INFO] Training epoch {e+1}/{num_epochs} completed.")

        mean_iou = 0
        iou_per_class = np.zeros(4)
        try:
            mean_iou, iou_per_class, confusion_mat = calc_miou(unet, testLoader)
            if mean_iou is not None:
                miou_per_epoch.append(mean_iou)
                iou_per_class_history.append(iou_per_class)
            else:
                miou_per_epoch.append(0)
                iou_per_class_history.append(np.zeros(4))
        except Exception as ex:
            print(f"[ERROR] mIoU calculation failed for epoch {e+1}: {ex}")
            miou_per_epoch.append(0)
            iou_per_class_history.append(np.zeros(4))

        print(f"[INFO] EPOCH: {e+1}/{num_epochs}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Mean IoU: {mean_iou}")
        for class_idx, class_name in enumerate(class_names):
            print(f"IoU for {class_name}: {iou_per_class[class_idx]}")

    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting the training losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='red', marker='o')
    plt.title('Training Loss on Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.svg'), format='svg')
    plt.close()

    # Plotting the mIoU per epoch
    plt.figure()
    plt.plot(range(1, num_epochs + 1), miou_per_epoch, label='Mean IoU', color='blue', marker='o')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch #')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_iou.svg'), format='svg')
    plt.close()
        
    # Save the model
    torch.save(unet.state_dict(), config.MODEL_PATH)
    print(f"[INFO] model saved to {config.MODEL_PATH}")

# Create the data loaders
trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

# Initialize the model
unet = UNet(in_channels=3, classes=4).to(config.DEVICE)

# Train the model and calculate mIoU
train_model_and_calc_miou(unet, trainLoader, testLoader, num_epochs=1, config=config, output_dir=output_dir)

# Function to display confusion matrix
def display_confusion_matrix(confusion_mat, output_dir):
    from sklearn.metrics import ConfusionMatrixDisplay
    confusion_matrix_svg_path = os.path.join(output_dir, 'confusion_matrix.svg')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=[str(i) for i in range(4)])
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_svg_path, format='svg')
    plt.close()

# Calculate mIoU on the test set and display the confusion matrix
mean_iou, iou_per_class, confusion_mat = calc_miou(unet, testLoader)
if confusion_mat is not None:
    display_confusion_matrix(confusion_mat, output_dir)

# Measure inference speed and peak memory usage
def measure_inference_speed(model, device, input_shape=(1, 3, config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), n_trials=200):
    model.eval().to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    durations = []
    memory_stats = []

    with torch.inference_mode():
        for _ in range(n_trials):
            start_event.record()
            outputs = model(dummy_input)
            predictions = torch.argmax(outputs, dim=1)
            predictions = predictions.byte().cpu()
            end_event.record()
            
            torch.cuda.synchronize()
            durations.append(start_event.elapsed_time(end_event))
            peak_memory = torch.cuda.max_memory_allocated()
            memory_stats.append(peak_memory)
    
    avg_inf_time = sum(durations) / len(durations)
    avg_memory = sum(memory_stats) / len(memory_stats)
    
    print(f"[INFO] Average Inference Time: {avg_inf_time} ms")
    print(f"[INFO] Average Peak Memory Usage: {avg_memory / (1024 ** 2)} MB")  # Convert to MB

# these are just for testing purposes for gathering data on the model and its performance, etc. not necessary for the actual model

# Measure the inference speed and peak memory usage
measure_inference_speed(unet, config.DEVICE)

# Load the saved model
model_path = '/home/mje2/projects/REU 2024/Transformers Stuff/Work with CAVs Dataset (CaT)/unetMLBD/output/unet_model.pth'

# Initialize the model architecture
unet_loaded = UNet(in_channels=3, classes=4)

# Load the saved model state dictionary
model_state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load the state dictionary into the model
unet_loaded.load_state_dict(model_state_dict)

# Move the model to the appropriate device
unet_loaded.to(config.DEVICE)

# Set the model to evaluation mode
unet_loaded.eval()

pytouch_total_params = sum(p.numel() for p in model.parameters()) 
print(f'{pytorch_total_params} params')

# Calculate the total number of parameters
total_params = sum(p.numel() for p in unet.parameters())

# Print the total number of parameters
print(f'Total number of parameters: {total_params:,} params')