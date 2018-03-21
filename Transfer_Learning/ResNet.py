import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time
import copy


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for data in dataloader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # ## Load Data
    # ---------

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder("tiny-imagenet-5/train/", data_transforms)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=25, shuffle=True, num_workers=4)

    dataset_size = 2500

    class_names = ['n09193705',
                   'n09246464',
                   'n09256479',
                   'n09332890',
                   'n09428293']

    use_gpu = torch.cuda.is_available()

    # Get a batch of training data
    inputs, classes = next(iter(dataloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    # ## ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad == False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.

    model_cnn = models.resnet34(pretrained=True)
    for param in model_cnn.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_cnn.fc.in_features
    model_cnn.fc = nn.Linear(num_ftrs, 5)

    if use_gpu:
        model_cnn = model_cnn.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of the final layer are being optimized
    optimizer_cnn = optim.Adam(model_cnn.fc.parameters())

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_cnn, step_size=5, gamma=0.1)

    model_cnn = train_model( model_cnn, criterion, optimizer_cnn, exp_lr_scheduler, num_epochs=1)
    torch.save( model_cnn.state_dict(), " model_cnn.pt")

    # ## Finetuning
    # Train the whole neural network. Only the last layer is changed

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters())

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)
    torch.save(model_ft.state_dict(), "model_ft.pt")
