#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt

if __name__=="__main__":
    X=list()
    Y=list()
    with open("error.log",'r') as f:
        for l in f:     
            x,y=map(float,l.split())
            X.append(x)
            Y.append(y)
    my_dpi=100
    plt.figure(figsize=(500/my_dpi,500/my_dpi),dpi=my_dpi)
    plt.xlabel("epoch")
    plt.ylabel("error (%)")
    plt.semilogx(X,Y,basex=2)
    print(X,Y)
    plt.savefig("error.pdf",dpi=my_dpi)

