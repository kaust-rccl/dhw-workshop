import argparse,os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models

def add_argument():
    parser = argparse.ArgumentParser(description="Tinyimagenet")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="number of dataloader cpus (default: 4)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Select a model
    model = models.resnet50()

   ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################

    # Load training data from TinyImageNet.
    trainset = datasets.ImageFolder("/ibex/reference/CV/tinyimagenet/train",
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    trainloader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    # Load validation data from TinyImageNet.
    valset = datasets.ImageFolder("/ibex/reference/CV/tinyimagenet/val",
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))

    valloader = torch.utils.data.DataLoader(valset, 
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
  
    ########################################################################
    # Step 2. Setup optimzer and loss funciton on GPU if using it
    ########################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

    ########################################################################
    # Step 3. Train the network.
    ########################################################################
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward(loss)
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % args.log_interval == (
                args.log_interval - 1):  # Print every log_interval mini-batches.
                print(
                    f"[Epoch: {epoch+1:5d}, batch:{i+1:5d}] Training loss: {running_loss / args.log_interval : .3f}"
                )
                running_loss = 0.0

        # Validation 
        running_loss_v  = 0.0
        model.eval()
        with torch.no_grad():
            for ii, data_v in enumerate(valloader):
                inputs_v,labels_v = data_v[0].to(device), data_v[1].to(device)
                outputs_v = model(inputs_v)
                loss_v  = criterion(outputs_v, labels_v)
                running_loss_v  += loss_v.item()
            print(
                f"[Epoch: {epoch + 1 :5d}]       Validation loss: {running_loss_v / ii  : .3f}"
            )
            
    PATH = './model-%s.pth' %epoch
    torch.save(model.state_dict(), PATH)

    print("Finished Training")

if __name__ == "__main__":
    args = add_argument()
    main(args)
