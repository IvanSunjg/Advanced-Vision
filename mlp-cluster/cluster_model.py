#imports
from avgmentations import losses
from avgmentations.resnet_dataset import ResNetImageFolder, RESNET_MEAN, RESNET_STD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms, datasets
import time
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from arg_extractor import get_args
from copy import copy
import experiment

#splits the train data to train and validation sets
def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    datasets['train'].dataset = copy(dataset)
    datasets['train'].dataset.transform = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize(RESNET_MEAN, RESNET_STD)
    ])
            
    datasets['val'].dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    return datasets

#### TRAIN FUNCTION ####
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, checkpoint=None):
    since = time.time()

    best_acc = 0.0
    start_epoch = 0
    if checkpoint:
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        since -= checkpoint['time_elapsed']

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        image_datasets['train'].dataset.update_transform(epoch)

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
                print(f'Saving checkpoint for epoch {epoch}...')
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'state_dict': best_model_wts, 
                    #'optimizer': optimizer.state_dict(), 
                    #'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 
                    'acc': best_acc,
                    'time_elapsed': time.time() - since
                }, 
                f'{path}/{exp_type}-{job_id}.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

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
    batch_size = args.batch_size
    num_workers = args.num_workers

    checkpoint_file = args.checkpoint_file
    job_id = args.job_id
    perform_inference = args.perform_inference

    exp_type = args.exp_type
    exp_kwargs = args.exp_kwargs

    #custom augmentations
    exp = experiment.Experiment(root=f'{path}/train', n_classes=1000)
    dataset = ResNetImageFolder(
        root=f'{path}/train',
        epoch_transforms=exp.construct_experiment(exp_type, **exp_kwargs)
    )

    print(args)

    #gets separated data
    image_datasets = train_val_dataset(dataset)
    #print the number of total images/train images/val images
    print(len(image_datasets['train']) + len(image_datasets['val']) )
    print(len(image_datasets['train']))
    print(len(image_datasets['val']))
    
    #dataloader shuffles both train and valid!!! image_datasets
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) 
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
    if checkpoint_file is None:
        FILE = f'{path}/resnet50_fconv_model_best.pth.tar'
    else:
        FILE = f'{path}/{checkpoint_file}'
    print(f'Loading {FILE}...')
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

    if perform_inference: # Inference on Test Set
        model.eval()

        test_dataset = datasets.ImageFolder(
            root=f'{path}/test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

        test_acc = 0.0
        for samples, labels in test_loader:
            with torch.no_grad():
                samples, labels = samples.cuda(), labels.cuda()
                output = model(samples)
                
                # calculate accuracy
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                test_acc += torch.mean(correct.float())
        print(f'Accuracy of the network on {len(test_dataset)} test images: {test_acc.item() / len(test_loader)}')

    else:
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
        model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, checkpoint=checkpoint)
