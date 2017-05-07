import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derSigmoid(x):
    return np.exp(x) / np.power((np.exp(x)+1),2)

def criaMatrizPesosDefault(linhas,colunas):
	return np.random.random((linhas,colunas)) - 1

def MLP(entrada,taxaDeAprendizado,epocas,erroMaximo,nroNeuronios,target):
    
    if len(entrada.shape) == 1:
        entrada.shape = (entrada.shape[0],1)
    if len(target.shape) == 1:
        target.shape = (target.shape[0],1)

    X = np.copy(np.transpose(entrada.copy()))

    T = np.copy(np.transpose(target.copy()))

    np.random.seed(1)
    #semente
    
    #preenchimento
    #checar regra
    v = criaMatrizPesosDefault(X.shape[1],nroNeuronios)
    w = criaMatrizPesosDefault(nroNeuronios,T.shape[1])

    print("v.shape = " + str(v.shape))
    print("w.shape = " + str(w.shape))

    for epoca in xrange(epocas):
        
        #calculo dos valores e ativação
        Z = sigmoid(np.dot(X,v))
        Y = sigmoid(np.dot(Z,w))

        #erro (diferença entre o target)
        taxaDeErroSaida = T - Y
        
        if (epoca % 10000) == 0:
            print "Error:" + str(np.mean(np.abs(taxaDeErroSaida)))
            
        #taxa de erro para os pesos w (∆w[j][k])
        deltaY = taxaDeErroSaida * derSigmoid(Y)

        #
        erroZ = deltaY.dot(w.T)
        
        # in what direction is the target Z?
        # were we really sure? if so, don't change too much.
        deltaZ = erroZ * derSigmoid(Z)

        #w[j][k](new) = w[j][k](old) + ∆w[j][k]
        w += Z.T.dot(deltaY)

        #v[i][j](new) = v[i][j](old) + ∆v[i][j]
        v += X.T.dot(deltaZ)

    return Y

# MAIN

'''
#problema do XOR:
entrada = np.array( [  [0., 0., 1.],
                       [0., 1., 1.], 
                       [1., 0., 1.], 
                       [1., 1., 1.] ])

saidaEsperada = np.array( [ [0.],
                            [1.],
                            [1.],
                            [0.]] )
'''

arquivo = "/home/mint/Documents/GIT/estudosIA/python/train_5a_00000.txt"
'''
53 = S
58 = X
5a = Z
'''
print("\tProcessando arquivo " + arquivo)
    
entrada = np.loadtxt(arquivo, dtype='float', delimiter="\n")
entrada = np.transpose(entrada) #transpoe de uma matriz linha para uma matriz coluna

print(entrada.shape)

alfa = 0.4

epocas = 60000

erroMaximo = 0.15

nroNeuronios = 15

'''
Definicoes da saida:
53 = S ==> saida esperada = (1, 0, 0)
58 = X ==> saida esperada = (0, 1, 0)
5a = Z ==> saida esperada = (0, 0, 1)
'''

saidaEsperada = np.array( [ [0.],
                            [0.],
                            [1.]] )

saida = MLP(entrada,alfa,epocas,erroMaximo,nroNeuronios,saidaEsperada)

print("")
print("Saida = ")
print(saida)