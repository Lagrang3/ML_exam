#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt

if __name__=="__main__":
    mat = list()
    with open("confusion.log",'r') as f:
        for l in f: 
            l=l.replace('[','').replace(']','').split()
            mat += map(float,l)
        
        mat = numpy.array(mat).reshape(16,16)
        #print(mat)
    
    my_dpi=100
    bird_labels  = ['amerob', 'barswa', 'blujay', 'carwre', 'comrav', 'comter', 
    'comyel', 'grhowl', 'houspa', 'houwre', 'mallar3', 'norcar', 
    'redcro', 'rewbla', 'sonspa', 'spotow']
    plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.yticks(numpy.arange(len(bird_labels)),labels=bird_labels)
    plt.xticks(numpy.arange(len(bird_labels)),labels=bird_labels,rotation=45)
    plt.imshow(mat)
    plt.colorbar() 
    plt.savefig("confusion.pdf",dpi=my_dpi)
