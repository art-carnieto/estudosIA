import numpy as np
import os
import sys
import matplotlib.pyplot as plt


dic2 = {(1) : 'A',
        (2) : 'Z'}

t1 = (1,2,3,4,5,3,0.4,5.1,0,1)
t1 = np.asarray(t1)
def teste():

    x = np.arange(0, 5, 0.1);
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(os.path.join(os.getcwd(), "teste1"))

def main():
    
    print np.argmax(t1)
    
    return
    
if __name__ == "__main__":
    main()