#coding: utf-8
import numpy as np
import os
from random import shuffle
import time
import datetime

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derSigmoid(x):
    return np.exp(x) / np.power((np.exp(x)+1),2)

def criaMatrizPesosDefault(linhas,colunas):
	return np.random.random((linhas,colunas)) - 1

def MLP(entrada,taxaDeAprendizado,epocas,erroMaximo,nroNeuronios,target, pesosV=None, pesosW=None):
    
    if len(entrada.shape) == 1:
        entrada.shape = (entrada.shape[0],1)
    if len(target.shape) == 1:
        target.shape = (target.shape[0],1)

    X = np.copy(np.transpose(entrada.copy()))

    T = np.copy(np.transpose(target.copy()))

    np.random.seed(1)
    #semente
    
    if pesosV is None:
        v = criaMatrizPesosDefault(X.shape[1],nroNeuronios)
    else:
        v = pesosV
    if pesosW is None:
        w = criaMatrizPesosDefault(nroNeuronios,T.shape[1])
    else:
        w = pesosW

    for epoca in xrange(epocas):
        
        #calculo dos valores e ativação
        Z_in = np.dot(X,v)

        Z = sigmoid(Z_in)

        Y_in = np.dot(Z,w)

        Y = sigmoid(Y_in)

        #erro (diferença entre o target)
        taxaDeErroSaida = T - Y
        
        #if (epoca % 10000) == 0:
            #print "Error:" + str(np.mean(np.abs(taxaDeErroSaida)))
            
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

    return Y, v, w

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

try:
    # caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo hog.py
    caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/HOG 32" # pasta selecionada pelo usuario
    arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
    print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

if len(arquivosImagem) == 0:
    print("Pasta selecionada nao contem imagens .txt!")

ordenados = sorted(arquivosImagem)

arquivos = ordenados
log = open("log.txt", "w")

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
log.write("Inicio da execucao da rede em " + str(st) + "\n\n")

pesosV = None
pesosW = None
for arquivo in arquivos:
    log.write("\n\tProcessando arquivo " + arquivo + "\n")
        
    entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
    entrada = np.append(entrada, [1.])
    entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

    alfa = 0.4

    epocas = 1000

    erroMaximo = 0.15

    nroNeuronios = 15

    '''
    Definicoes da saida:
    53 = S ==> saida esperada = (1, 0, 0)
    58 = X ==> saida esperada = (0, 1, 0)
    5a = Z ==> saida esperada = (0, 0, 1)
    '''
    letra = None
    if "_53_" in arquivo:
        saidaEsperada = np.asarray(dic1['53'])
        letra = "S"
    elif "_58_" in arquivo:
        saidaEsperada = np.asarray(dic1['58'])
        letra = "X"
    elif "_5a_" in arquivo:
        saidaEsperada = np.asarray(dic1['5a'])
        letra = "Z"
    else:
        log.write("\nERRO: arquivo de entrada nao eh 'S', nem 'X' nem 'Z'! (nome errado ou alterado)\n")

    saida, pesosV, pesosW = MLP(entrada,alfa,epocas,erroMaximo,nroNeuronios,saidaEsperada, pesosV, pesosW)

    log.write("\nSaida = \n")
    log.write(str(saida))

    log.write("\n\nSaida arredondada = \n")
    saidaArredondada = []
    for i in range(saida.size):
        saidaArredondada.append(round(saida[0][i], 0))
    log.write(str(saidaArredondada))
    log.write("\n\nLetra da entrada = " + letra)
    try:
        log.write("\nSaida em letra = " + str(dic2[tuple(saidaArredondada)]))
        if letra == dic2[tuple(saidaArredondada)]:
            log.write("\nAcertou!\n\n")
        else:
            log.write("\n@@@@@@@@@@ERRO@@@@@@@@@@\n\n")
    except KeyError as err:
        log.write("\nERRO: saida nao corresponde a nenhum caractere!\n")
        log.write("\nSaida = " + str(saidaArredondada) + "\n")

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
log.write("Fim da execucao da rede em " + str(st) + "\n\n")