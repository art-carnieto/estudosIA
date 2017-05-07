#coding: utf-8
import numpy as np
import os

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
        Z_in = np.dot(X,v)

        Z = sigmoid(Z_in)

        Y_in = np.dot(Z,w)

        Y = sigmoid(Y_in)

        #erro (diferença entre o target)
        taxaDeErroSaida = T - Y
        
        if (epoca % 10000) == 0:
            print "Error:" + str(np.mean(np.abs(taxaDeErroSaida)))
            
        #taxa de erro para segunda camada de pesos (δw[k])
        taxaDeErroW = taxaDeErroSaida * derSigmoid(Y_in)
        
        #∆w[j][k] = α*δw[k]*Z[j]
        deltaW = taxaDeAprendizado * taxaDeErroW * np.transpose(Z)

        #erro para V (δv_inv[j] = ∑ k=1 δw[k]w[j][k] )
        taxaDeErroEscondida = taxaDeErroW.dot(np.transpose(w))

        # δv[j] = δv_in[j] f′(z_in[j])
        taxaDeErroV = taxaDeErroEscondida * derSigmoid(Z_in)

        #∆v[i][j] = αδ[j]
        deltaV = taxaDeAprendizado * np.transpose(X) * taxaDeErroV

        #w[j][k](new) = w[j][k](old) + ∆w[j][k]
        w += deltaW

        #v[i][j](new) = v[i][j](old) + ∆v[i][j]
        v += deltaV

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

dic1 = {'53': (1,0,0), '58': (0,1,0), '5a': (0,0,1)}    # dicionario 1 ==> para gerar a matriz T (target)
dic2 = {(1,0,0) : 'S', (0,1,0) : 'X', (0,0,1) : 'Z'}    # dicionario 2 ==> para verificar depois de rodar a rede a resposta

pasta = "/home/mint/Documents/GIT/estudosIA/python"
arquivo = "train_5a_00000.txt"
print("\tProcessando arquivo " + arquivo)
    
entrada = np.loadtxt(os.path.join(pasta, arquivo), dtype='float', delimiter="\n")
entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

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
'''
saidaEsperada = np.array( [ [0.],
                            [0.],
                            [1.]] )
'''

if "53" in arquivo:
    saidaEsperada = np.asarray(dic1['53'])
elif "58" in arquivo:
    saidaEsperada = np.asarray(dic1['58'])
elif "5a" in arquivo:
    saidaEsperada = np.asarray(dic1['5a'])
else:
    print("ERRO: arquivo de entrada nao eh 'S', nem 'X' nem 'Z'! (nome errado ou alterado)")

saida = MLP(entrada,alfa,epocas,erroMaximo,nroNeuronios,saidaEsperada)

print("")
print("Saida = ")
print(saida)


print("")
print("Saida arredondada = ")
saidaArredondada = []
for i in range(saida.size):
    saidaArredondada.append(round(saida[0][i], 0))
    print(saidaArredondada[i])


print("")
print("Saida em letra = " + str(dic2[tuple(saidaArredondada)]))
