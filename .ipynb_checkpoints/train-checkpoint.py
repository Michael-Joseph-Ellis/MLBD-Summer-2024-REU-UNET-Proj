# python train.py
# import the necessary packages
import torchvision
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
from pyimagesearch.dataset2 import SegmentationDataset
from pyimagesearch.model3 import UNet
from pyimagesearch import config
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam
import torchvision.transforms as transforms
from imutils import paths
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix

class SegmentationDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        #loop through each location
        #Go into the split folder
        #Go into images and masks since they will be in pairs
        
        # store the image and mask filepaths, and augmentation
        # transforms
        self.root = root
        
        self.transforms = transforms
        
        #split is included b/c the data has a train, val, and test folder 
        #ex: urRootPath/Brown_Field/Train/annos/int_maps/mask_101.png  (mask in train folder)
        #ex: urRootPath/ (img in val folder)
        
        #stores the image paths and mask paths (duh)
        self.image_paths = []
        self.mask_paths = []
        
        #weird, int_map masks in brown_field is named mask_numberHere.png while in Powerline its anno_pln_numberHere.png
        for trail_name in os.listdir(self.root):
            split_path = os.path.join(root, trail_name, split.capitalize())
            imgs_path = os.path.join(split_path, "imgs")
            masks_path = os.path.join(split_path, "annos/int_maps/")
            for mask_name in os.listdir(masks_path):
                    
                    #Get the int_map mask first
                    mask_path = os.path.join(masks_path, mask_name)
                    self.mask_paths.append(mask_path)
            
                    # get the corresponding image to that map_mask
                    image_name = "img_" + mask_name.split("_", 1)[1] # need to change file name a little for images
                    image_path = os.path.join(imgs_path, image_name)
                    self.image_paths.append(image_path)
        
        # the images are separated in different folders by trail so we need to loop through those
        # both images and masks have the same trail (b/c image must have a corresponding mask) so we only loop through that once

                    
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image, mask = self.transforms(image, mask)
        # return a tuple of the image and its mask
        return (image, mask)
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        
        #Numpy Array stores image data as [Height, Width, Num_Color_Channels]
        #Pytorch Tensor stores image data as [Num_Color_Channels, Height, Width]

        # we cannot use transforms.ToTensor() for masks as it normalizes the data between 0.0 and 1.0 which we dont want
        # so instead we first convert to numpy array and then to tensor of type int 64 to avoid this
        # thus our integer labels in the image data is preserved
        
        # mask
        target = np.array(target)
        target = torch.tensor(target, dtype=torch.int64) #int64 b/c target (in this case masks) have to be int64 for crossEntropyLoss

        # however we do want to use ToTensor for images to normalize to help prevent giving too high of an initial value to color values
        # image
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        
        return image, target

# WARNING: Do not use Transform.ToTensor as it normalizes the data [0.0-1.0] which again, we don't want
# aug apply to mask and img
transform = []
# WARNING: Interpolation nearrest is needed to prevent the pixels in mask from being blurred when stretched thus messing up the labelIds
transform.append( transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST )) 
transform = Compose(transform)

trainDS = SegmentationDataset('/home/mje2/projects/REU 2024/Transformers Stuff/CaT/CAT', split='Train',
	transforms=transform)
testDS = SegmentationDataset('/home/mje2/projects/REU 2024/Transformers Stuff/CaT/CAT', split='Test',
	transforms=transform)


# initialize our UNet model
unet = UNet(in_channels=3, classes=4).to(config.DEVICE)
# initialize loss function and optimizer
#0 is the index for everything that isn't the traversable terrain
lossFunc = CrossEntropyLoss() 
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# function to calculate mIoU
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

    if not predictions or not targets:
        print("[ERROR] No predictions or targets were collected.")
        return None, None, None

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0).astype(np.int64)

    targets = targets.flatten()
    predictions = predictions.flatten()

    if targets.size == 0 or predictions.size == 0:
        print("[ERROR] Flattened targets or predictions are empty.")
        return None, None, None

    confusion_mat = confusion_matrix(targets, predictions)
    iou_per_class = np.diag(confusion_mat) / (confusion_mat.sum(axis=1) + confusion_mat.sum(axis=0) - np.diag(confusion_mat))
    mean_iou = np.nanmean(iou_per_class)

    print(f"[DEBUG] Mean IoU: {mean_iou}")
    print(f"[DEBUG] IoU per class: {iou_per_class}")
    print(f"[DEBUG] Confusion Matrix: {confusion_mat}")

    return mean_iou, iou_per_class, confusion_mat

def train_model_and_calc_miou(unet, trainLoader, testLoader, num_epochs, config):
    train_losses = []
    miou_per_epoch = []

    # initialize the optimizer and loss function
    lossFunc = CrossEntropyLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)
    trainSteps = len(trainLoader)
    testSteps = len(testLoader)

    print("[INFO] training the network...")
    startTime = time.time()
    for e in range(num_epochs):
        # set the model in training mode
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

        print(f"[DEBUG] Training epoch {e+1}/{num_epochs} completed. Calculating mIoU...")
        
        # initialize mean_iou to handle cases where calc_miou might fail
        mean_iou = 0
        try:
            mean_iou, iou_per_class, confusion_mat = calc_miou(unet, testLoader)
            if mean_iou is not None:
                miou_per_epoch.append(mean_iou)
            else:
                print("[ERROR] mIoU calculation returned None for epoch:", e + 1)
                miou_per_epoch.append(0)
        except Exception as ex:
            print(f"[ERROR] mIoU calculation failed for epoch {e+1}: {ex}")
            miou_per_epoch.append(0)

        print(f"[INFO] EPOCH: {e+1}/{num_epochs}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Mean IoU: {mean_iou}")

    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")
    
    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # plotting the training losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='red', marker='o')
    plt.title('Training Loss on Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # plotting the mIoU per epoch
    plt.figure()
    plt.plot(range(1, num_epochs + 1), miou_per_epoch, label='Mean IoU', color='blue', marker='o')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch #')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_iou.png'))
    plt.close()

# create the data loaders
trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

# initialize the model
unet = UNet(in_channels=3, classes=4).to(config.DEVICE)

output_dir = "/home/mje2/projects/REU 2024/Transformers Stuff/Work with CAVs Dataset (CaT)/unetMLBD/output"

# train the model and calculate mIoU for each epoch
train_model_and_calc_miou(unet, trainLoader, testLoader, num_epochs=100, config=config)

# function to display confusion matrix
def display_confusion_matrix(confusion_mat, output_dir):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=[str(i) for i in range(4)])
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# calculate and display confusion matrix after training
mean_iou, iou_per_class, confusion_mat = calc_miou(unet, testLoader)
if confusion_mat is not None:
    display_confusion_matrix(confusion_mat, output_dir)
