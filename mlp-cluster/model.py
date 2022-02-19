#imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from copy import copy
from arg_extractor import get_args
from copy import copy
#reads the path to files
args = get_args()
path = args.path
# mean and std of full Imagenet dataset
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


#splits the train data to train and validation sets
def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    datasets['train'].dataset = copy(dataset)
    #these train and val basic augmentation are done at loading images 
    #it's data preprocessing don't modify it
    datasets['train'].dataset.transform = transforms.Compose([
    
    	transforms.ToTensor(),
    	transforms.Normalize(mean, std)
            ])
            
    datasets['val'].dataset.transform = transforms.Compose([
 		
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
            ])
    return datasets

#loads the train folder
dataset = ImageFolder(path+'/train')

#gets separated data
image_datasets = train_val_dataset(dataset)
#print the number of total images/train images/val images
print(len(image_datasets['train']) + len(image_datasets['val']) )
print(len(image_datasets['train']))
print(len(image_datasets['val']))

#### RANDOM AUGMENTATION ####
#this is what you should modify
# whatewer transforms you include in the Compose will be executed during training at each batch
# best val accuracy with these augmentations is 35%
image_datasets['train'].dataset.transform = transforms.Compose([

	transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
            ])
                      
#dataloader shuffles both train and valid!!! image_datasets
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=8)
              for x in ['train','val']}
              
#full dataset size needed for training              
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}

#declaring the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#to see the dataset images 
def imshow(inp, title):
    """Imshow for Tensor."""
    
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()
    
#### TRAIN FUNCTION ####
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

#    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
 		
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #out = torchvision.utils.make_grid(inputs)
            #imshow(out, title=['train'+str(x.item()) for x in labels])
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
#    model.load_state_dict(best_model_wts)
    return model


#### Finetuning the model ####
# Load a pretrained model and reset50
      
model = models.resnet50(pretrained=False)
#freeze first two layers
ct = 0
for child in model.children():
	ct += 1
	if ct < 3:
		for param in child.parameters():
    			param.requires_grad = False

#load checkpoint
FILE = path+"/resnet50_fconv_model_best.pth.tar"
checkpoint = torch.load(FILE)

model = nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])



#modify fully connected layers
'''
num_ftrs = model.module.fc.in_features
model.module.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),nn.ReLU(),nn.Dropout(0.5),model.module.fc)
'''
#print layers
'''
for child in model.children():
	print(child)
'''
# hyperparameters 
#these give the best result so far 35%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.006, weight_decay=0.006)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

model = model.to(device)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=15)



