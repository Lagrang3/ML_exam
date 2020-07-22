#!/usr/bin/env python3

from scipy.stats import binom
import math
if __name__=="__main__":
    N=312
    p=1/16
    frac=0.5431
    #frac=15/16 guessing
    x=int((1-frac)*N)
    print("Probability i am guessing:",binom.logsf(x,N,p)/math.log(10))
    
