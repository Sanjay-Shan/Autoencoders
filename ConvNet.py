import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        #model_1
        self.encoder1 = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            )
        self.decoder1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256,784),
            nn.Sigmoid()
            )

        # model_2
        self.encoder_decoder = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3,padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(True),
        nn.Conv2d(4, 8, kernel_size=3,padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(True),
        # nn.Upsample(size=(8,7,7) ,mode='bilinear'),
        nn.ConvTranspose2d(8,4,kernel_size=3, stride=1,padding=1),
        nn.Upsample(size=(14,14), mode='bilinear'),
        nn.ConvTranspose2d(4, 2, kernel_size=3, stride=1,padding=1),
        nn.Upsample(size=(28,28), mode='bilinear'),
        nn.ConvTranspose2d(2,1, kernel_size=3, stride=1,padding=1),
        nn.Sigmoid()
        )

        

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        #forward feed
        x = X.view(X.shape[0], -1)
        x=self.encoder1(x)
        x=self.decoder1(x)
        return x

    def model_2(self,X):
        x=self.encoder_decoder(X)
        return x

    
            
    
