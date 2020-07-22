import torch
import torch.nn.functional as F
import copy, numpy

ch1 = 16
ch2 = 32
ch3 = 64
feat1 = 500
feat2 = 100

class Oracle(torch.nn.Module):
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        super(Oracle,self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = ch1 , 
            kernel_size = 3,padding=1) # ?
        self.conv2 = torch.nn.Conv1d(in_channels = ch1, out_channels = ch2,
            kernel_size =3,padding=1) # ?
        self.conv3 = torch.nn.Conv1d(in_channels = ch2, out_channels = ch3,
            kernel_size =3,padding=1) # ?
        
        self.fc1 = torch.nn.Linear(in_features = ch3*input_size//8,out_features = feat1)
        self.fc2 = torch.nn.Linear(in_features = feat1 ,out_features = feat2)
        self.fc3 = torch.nn.Linear(in_features = feat2 ,out_features = output_size)
   
    
    def forward(self,x):
        x = F.relu( F.max_pool1d( self.conv1(x),2))
        
        x = F.relu( F.max_pool1d( self.conv2(x),2))
        
        x = F.relu( F.max_pool1d( self.conv3(x),2))
        
        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        #x = F.dropout(x,training=self.training)
        
        x = F.relu(self.fc2(x))
        #x = F.dropout(x,training=self.training)
        
        #return torch.sigmoid( self.fc3(x))
        return torch.softmax( self.fc3(x),dim=1)
        
    def classify(self,x):
        x.unsqueeze_(1)
        y = self.forward(x)
        y = y.sum(0)
        y = torch.argmax(y).item()
        return y
    def detailed_classify(self,x):
        x.unsqueeze_(1)
        y = self.forward(x)
        print(y)
        y = y.sum(0)
        print(y)
        y = torch.argmax(y).item()
        print(y)
        return y
