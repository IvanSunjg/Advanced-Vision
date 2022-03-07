#imports
from avgmentations import augmentations, losses, utils
from avgmentations.resnet_dataset import ResNetImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from arg_extractor import get_args
from copy import copy

#splits the train data to train and validation sets
def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    datasets['train'].dataset = copy(dataset)
    
    datasets['train'].dataset.transform = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize(utils.RESNET_MEAN, utils.RESNET_STD)
    ])
            
    datasets['val'].dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(utils.RESNET_MEAN, utils.RESNET_STD),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    return datasets

#### TRAIN FUNCTION ####
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        image_datasets['train'].update_transform(epoch)

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
                #running_corrects += torch.sum(preds == labels.data)
                targets = torch.max(labels.data, dim=1).indices
                running_corrects += torch.sum(preds == targets)
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
            # best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    ###command line rguments###
    args = get_args()
    path = args.path
    num_epochs = args.num_epochs
    optimizer_type = args.optimizer_type
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    beta1 = args.beta1
    beta2 = args.beta2
    amsgrad = args.amsgrad
    loss_function = args.loss_function
    #reduction = args.reduction
    #label_smoothing = args.label_smoothing
    step_size = args.step_size
    gamma = args.gamma
    num_of_frozen_blocks = args.num_of_frozen_blocks 
    argument1 = args.argument1
    argument2 = args.argument2
    argument3 = args.argument3
    argument4 = args.argument4
    argument5 = args.argument5
    argument6 = args.argument6
    argument7 = args.argument7

    default_dataset = ImageFolder(
        root = f'{path}/train',
        transform = utils.DEFAULT_AUGMENTATION,
        target_transform = augmentations.OneHot(1000)
    )

    #custom augmentations
    dataset = ResNetImageFolder(
        f'{path}/train',
        {
            0: augmentations.Compose([
                transforms.ToTensor(),
                transforms.Normalize(utils.RESNET_MEAN, utils.RESNET_STD),
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),

                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(), 
                #transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomAdjustSharpness(sharpness_factor=2),
                #transforms.RandomAutocontrast(),
                
                #augmentations.Double(argument1), # Custom Double Augmentation
                #augmentations.AugMix()
                #augmentations.CornerMask(argument1)
                augmentations.CutOut(argument1,argument2), # Custom Cutout Augmentation
                #augmentations.GridMask(), # Custom GridMask Augmentation
                #augmentations.MixUp(default_dataset, alpha=argument1, min_lam=argument2, max_lam=argument3),
                #augmentations.CutMix(default_dataset, alpha=argument1, min_lam=argument2, max_lam=argument3),
                #augmentations.P(augmentations.AugMix(), p=argument1) # Apply AugMix with 'argument1' probability
                #augmentations.OneOf([augmentations.AugMix(), augmentations.GridMask()], p=[0.3, 0.7]) # Choose either AugMix or GridMask with probabilities 0.3 and 0.7 respectively. Can choose from any number of transformations
            ])
        }
    )

    #gets separated data
    image_datasets = train_val_dataset(dataset)
    #print the number of total images/train images/val images
    print(len(image_datasets['train']) + len(image_datasets['val']) )
    print(len(image_datasets['train']))
    print(len(image_datasets['val']))
    
    #dataloader shuffles both train and valid!!! image_datasets
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=1) 
        for x in ['train','val']
    }
                
    #full dataset needed for training              
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}

    #declaring the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #### Finetuning the model ####
    # Load a pretrained model and reset final fully connected layer.
    model = models.resnet50(pretrained=False)

    #load checkpoint
    #FILE = "/media/gabriel/24C755C11481A4EA/resnet50_fconv_model_best.pth.tar"
    FILE = f'{path}/resnet50_fconv_model_best.pth.tar'
    checkpoint = torch.load(FILE)
    #froze blocks
    ct = 0
    for child in model.children():
        ct += 1
        if ct < num_of_frozen_blocks+1:
            for param in child.parameters():
                param.requires_grad = False
    #enables batch layers in frozen block			
    for name, param in model.named_parameters():
        if 'bn' in name:
            param.requires_grad = True

    model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['state_dict'])

    criterion = losses.CrossEntropyLoss()
    if loss_function == 'BCE':
        criterion = losses.BCEntropyLoss()
    '''
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction=reduction)
    if loss_function=='KLD':
        criterion = nn.KLDivLoss(reduction=reduction)
    '''


    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=amsgrad)

    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = model.to(device)
    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs)
