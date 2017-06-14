import numpy as np
import os
import sys
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1);
y = np.sin(x)
plt.plot(x, y)

def teste():

    x = np.arange(0, 5, 0.1);
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(os.path.join(os.getcwd(), "teste1"))

def main():
    
    teste()

    return
    
if __name__ == "__main__":
    main()