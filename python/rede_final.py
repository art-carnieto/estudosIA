import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derSigmoid(x):
    return np.exp(x) / np.power((np.exp(x)+1),2)

def MLP(entrada,taxaDeAprendizado,epocas,erroMaximo,target,nroNeuronios):
    
    X = np.copy(entrada)

    T = np.copy(target)

    np.random.seed(1)
    #semente
    
    # randomly initialize our weights with mean 0
    #preenchimento
    #checar regra
    v = 2*np.random.random((X.shape[1],nroNeuronios)) - 1
    w = 2*np.random.random((nroNeuronios,T.shape[1])) - 1

    for epoca in xrange(epocas):

    	# Feed forward through layers 0, 1, and 2
        
        Z = sigmoid(np.dot(X,v))
        Y = sigmoid(np.dot(Z,w))

        # how much did we miss the target value?
        taxaDeErroSaida = T - Y
        
        if (epoca % 10000) == 0:
            print "Error:" + str(np.mean(np.abs(taxaDeErroSaida)))
            
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        deltaY = taxaDeErroSaida * derSigmoid(Y)

        # how much did each Z value contribute to the Y error (according to the weights)?
        erroZ = deltaY.dot(w.T)
        
        # in what direction is the target Z?
        # were we really sure? if so, don't change too much.
        deltaZ = erroZ * derSigmoid(Z)

        w += Z.T.dot(deltaY)
        v += X.T.dot(deltaZ)

    return Y

# MAIN


#problema do XOR:
entrada = np.array( [  [0., 0., 1.],
                       [0., 1., 1.], 
                       [1., 0., 1.], 
                       [1., 1., 1.] ])

saidaEsperada = np.array( [ [0.],
                            [1.],
                            [1.],
                            [0.]] )

alfa = 0.4

epocas = 60000

erroMaximo = 0.15

nroNeuronios = 15

saida = MLP(entrada,alfa,epocas,erroMaximo,saidaEsperada,nroNeuronios)

print("")
print("Saida = ")
print(saida)