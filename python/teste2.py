import numpy as np
import os
import sys
import matplotlib.pyplot as plt


dic2 = {(1) : 'A',
        (2) : 'Z'}

t1 = (
        (1,0,0),
        (3,2,0),
        (3,0,4)
     )
     
t1 = np.asarray(t1)

def acuracia(x):
    
    count = 0
    total = 0
    
    
    if(x.shape[0] == x.shape[1]):
        for i in range(x.shape[0]):
            count += x[i][i]
            
    for j in range(x.shape[0]):
        for k in range(x.shape[1]):
            total += x[j][k]
    
    return count / float(total)
    
def main():
    
    print acuracia(t1)
    
    return
    
if __name__ == "__main__":
    main()