#coding: utf-8
import numpy as np
import os
from random import shuffle
import time
import datetime
from random import shuffle

np.random.seed(1)
    #semente

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derSigmoid(x):
    return np.exp(x) / np.power((np.exp(x)+1),2)

def erroQuadratico(x, T):
    return np.sum(np.power(T - x,2))/2

def criaMatrizPesosDefault(linhas,colunas):
	return np.random.random((linhas,colunas)) - 1

def MLP(entrada,taxaDeAprendizado,epocas,erroMaximo,nroNeuronios,target, pesosV=None, pesosW=None):
    
    if len(entrada.shape) == 1:
        entrada.shape = (entrada.shape[0],1)
    if len(target.shape) == 1:
        target.shape = (target.shape[0],1)

    X = np.copy(np.transpose(entrada.copy()))

    T = np.copy(np.transpose(target.copy()))

    if pesosV is None:
        v = criaMatrizPesosDefault(X.shape[1],nroNeuronios)
    else:
        v = pesosV
    if pesosW is None:
        w = criaMatrizPesosDefault(nroNeuronios,T.shape[1])
    else:
        w = pesosW

    taxaDeErroSaida = 0

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

    erroTreinamento = erroQuadratico(Y, T)

    return Y, v, w, erroTreinamento

def testaMLP(entrada, pesosV, pesosW):
    if len(entrada.shape) == 1:
        entrada.shape = (entrada.shape[0],1)

    X = np.copy(np.transpose(entrada.copy()))

    Z_in = np.dot(X,pesosV)

    Z = sigmoid(Z_in)

    Y_in = np.dot(Z,pesosW)

    Y = sigmoid(Y_in)

    return Y

# MAIN

config = open("config.txt", "w")
erro = open("erro.txt", "w")
model = open("model.dat", "wb")

global alfa
alfa = 0.4
global epocas
epocas = 1000
global erroMaximo
erroMaximo = 0.15
global nroNeuronios
nroNeuronios = 15
global extrator
extrator = "HOG"

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
print("Iniciando a execucao em " + st)
config.write("Execucao em " + st + "\n")
erro.write("Execucao em " + st + "\n")

dic1 = {'53': (1,0,0), '58': (0,1,0), '5a': (0,0,1)}    # dicionario 1 ==> para gerar a matriz T (target)
dic2 = {(1,0,0) : 'S', (0,1,0) : 'X', (0,0,1) : 'Z'}    # dicionario 2 ==> para verificar depois de rodar a rede a resposta

if extrator is "HOG":
    configText = [
        "extrator : HOG",
        "extrator_orientacoes: 10",
        "extrator_pixel_por_celula : 32",
        "extrator_celula_por_bloco : 1\n",
        "rede_alpha : " + str(alfa),
        "rede_camada_Z_neuronios : " + str(nroNeuronios),
        "rede_camada_Z_funcao_de_ativacao : sigmoide",
        "rede_camada_Y_neuronios : 3",
        "rede_camada_Y_funcao_de_ativacao : sigmoide",
        "rede_inicializacao_pesos : aleatoria",
        "rede_min_epocas : 0",
        "rede_max_epocas : " + str(epocas),
        "rede_parada_antecipada : loop que vai de 0 a max epocas"
    ]

elif extrator is "LBP":
    configText = [
        "extrator : LBP",
        "extrator_numero_pontos: 24",
        "extrator_raio : 8",
        "extrator_metodo : uniform\n",
        "rede_alpha : " + str(alfa),
        "rede_camada_Z_neuronios : " + str(nroNeuronios),
        "rede_camada_Z_funcao_de_ativacao : sigmoide",
        "rede_camada_Y_neuronios : 3",
        "rede_camada_Y_funcao_de_ativacao : sigmoide",
        "rede_inicializacao_pesos : aleatoria",
        "rede_min_epocas : 0",
        "rede_max_epocas : " + str(epocas),
        "rede_parada_antecipada : loop que vai de 0 a max epocas"
    ]

for i in configText:
    config.write(i + "\n")

try:
    # caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo hog.py
    if extrator is "HOG":
        caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/HOG 32" # pasta selecionada pelo usuario
    elif extrator is "LBP":
        caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/LBP" # pasta selecionada pelo usuario
    arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
    print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

if len(arquivosImagem) == 0:
    print("Pasta selecionada nao contem imagens .txt!")

ordenados = sorted(arquivosImagem)

arquivos = ordenados

listaS = ordenados[0:1000]
listaX = ordenados[1000:2000]
listaZ = ordenados[2000:3000]

fold1 = listaS[0:200] + listaX[0:200] + listaZ[0:200]
fold2 = listaS[200:400] + listaX[200:400] + listaZ[200:400]
fold3 = listaS[400:600] + listaX[400:600] + listaZ[400:600]
fold4 = listaS[600:800] + listaX[600:800] + listaZ[600:800]
fold5 = listaS[800:1000] + listaX[800:1000] + listaZ[800:1000]

listaFolds = [fold1, fold2, fold3, fold4, fold5]

for i in range(5):
    teste = listaFolds[i]
    treinamento = []
    for j in range(5):
        if(j != i):
            treinamento = treinamento + listaFolds[j]

    shuffle(treinamento) # embaralha a ordem dos arquivos de treinamento

    treinamento = ["train_5a_00000.txt"]
    somaTotalErro = 0
    pesosV = None
    pesosW = None
    for arquivo in treinamento:     # loop de treinamento dos folds atuais
            
        entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
        entrada = np.append(entrada, [1.])
        entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

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
            print("\nERRO: arquivo de entrada nao eh 'S', nem 'X' nem 'Z'! (nome errado ou alterado)\n")

        saida, pesosV, pesosW, erroTreinamento = MLP(entrada,alfa,epocas,erroMaximo,nroNeuronios,saidaEsperada, pesosV, pesosW)

        somaTotalErro = somaTotalErro + erroTreinamento

        saidaArredondada = []
        for i in range(saida.size):
            saidaArredondada.append(round(saida[0][i], 0))
        try:
            letraFinal = str(dic2[tuple(saidaArredondada)])
        except KeyError:
            letraFinal = None
        #if letra == letraFinal:
            #condicao de acerto
        #else:
            #condicao de erro

    erroFinal = somaTotalErro / len(treinamento)
    erro.write(str(i) + ";" + str(erroFinal) + ";0.0\n")

    '''
    for arquivo in teste:       # loop de teste dos folds
        entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
        entrada = np.append(entrada, [1.])
        entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

        saidaTestada = testaMLP(entrada, pesosV, pesosW)
    '''

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
print("Fim da execucao da rede em " + str(st) + "\n\n")