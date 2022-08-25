from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from prettytable import PrettyTable
# from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import argparse
import numpy as np 
import cv2
from torchvision.utils import save_image

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        if FLAGS.mode==1:
            data=data.view(data.shape[0], -1) #for mode=1

        loss = criterion(output,data)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch,FLAGS.num_epochs,loss.item()))
    
    

def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()

    criterion=nn.MSELoss()
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            if batch_idx==150:
                break
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            # if FLAGS.mode==1:
            #     output.resize(28,28)
            #     save_image(output,"mode1/"+str(batch_idx)+".jpg")
            # if FLAGS.mode==2:
            #     save_image(output,"mode2/"+str(batch_idx)+".jpg")
            
            output = output.detach().numpy() # for mode=1
            output = np.reshape(output,(28,28)) #only if mode==1
            # print(output.shape,type(output))
            
            cv2.imshow("test",output)
            cv2.waitKey(0)
            # if batch_idx==150:
            #     break
            

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print("cuda check done", use_cuda)
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    print("Initialize the model from the convnet",model)

    
    # Define loss function.
    criterion = nn.MSELoss()
    print("loss defined")
    
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    print("optimiser defined")
        
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = 1, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    print("data preparation done , moving towards training")
    f=open("model_"+str(FLAGS.mode)+"_output.txt","a")
    f.write("train_loss"+"   "+"train_accuracy"+"   "+"test_loss"+"   "+"test_accuracy)"+"\n")
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train(model, device, train_loader,optimizer, criterion, epoch, FLAGS.batch_size)
    output = test(model, device, test_loader)

    print("printing the number of parameters in the network")

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1  and 2.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.5,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    