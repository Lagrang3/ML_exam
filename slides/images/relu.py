#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy

if __name__=="__main__":
    x=torch.tensor(numpy.arange(-1,1,0.01))
    y=torch.nn.functional.relu(x)
    
    my_dpi=100
    plt.figure(figsize=(500/my_dpi,500/my_dpi),dpi=my_dpi)
    plt.xlabel("x")
    plt.ylabel("relu(x)")
    plt.plot(x,y)
    plt.title("ReLu")
    plt.savefig("relu.pdf",dpi=my_dpi)

