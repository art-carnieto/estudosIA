#coding: utf-8
import numpy as np
import os
from random import shuffle
import time
import datetime
from random import shuffle
import pickle
import sys

# np.random.seed(1) #semente de aleatoriedade

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

def main(argv):

    if(len(argv) != 6):
        print("Numero errado de argumentos!")
        print("Usagem do rede_final.py:")
        print("argumento-01: Pasta com a entrada da rede, deve comecar por 'HOG' ou 'LBP' (sem aspas)")
        print("argumento-02: Alfa (taxa de aprendizado) a ser usado na rede")
        print("argumento-03: Numero de epocas que sera usado no MLP")
        print("argumento-04: Erro maximo da rede MLP")
        print("argumento-05: Numero de neuronios da camada escondida da rede MLP")
        return

    if not(argv[1].startswith("HOG", 0, 3) or argv[1].startswith("LBP", 0, 3)):
        print("Extrator desconhecido!")
        print("O argumento-01 deve ser o nome da pasta com a entrada e deve comecar por 'HOG' ou 'LBP' (sem aspas), por exemplo: 'HOG', 'HOG1', 'HOG2', ...")
        return

    '''
    Testar depois se as variaveis realmente precisam ser globais:
    global extrator
    global alfa
    global epocas
    global erroMaximo
    global nroNeuronios
    '''
    
    extrator = str(argv[1])
    alfa = float(argv[2])
    epocas = int(argv[3])
    erroMaximo = float(argv[4])
    nroNeuronios = int(argv[5])

    caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/" # pasta selecionada pelo usuario

    try:
        caminhoEntrada = os.path.join(caminhoEntrada, extrator)
        arqExtrator = open(os.path.join(caminhoEntrada, "configExtrator.dat"), "rb")
    except IOError as err:
        print("Erro no acesso a pasta com as imagens de entrada: ",err)
        return

    i = 1
    existePasta = True
    try:
        while(existePasta == True):
            caminhoSaida = os.path.join(os.getcwd(),"execucao" + str(i))
            if os.path.exists(caminhoSaida):
                i += 1
            else:
                existePasta = False
        os.makedirs(caminhoSaida)
    except OSError as err:
        print("Erro de acesso a pasta de saida: ", err)
        return

    try:
        config = open(os.path.join(caminhoSaida,"config.txt"), "w")
        erro = open(os.path.join(caminhoSaida,"erro.txt"), "w")
        model = open(os.path.join(caminhoSaida,"model.dat"), "wb")
    except IOError as err:
        print("Erro na escrita dos arquivos de saida", err)
        return

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
    print("Iniciando a execucao em " + st)
    config.write("Execucao em " + st + "\n")
    erro.write("Execucao em " + st + "\n")

    confExt = pickle.load(arqExtrator)
    
    if extrator.startswith("HOG", 0, 3):
        configText = [
            "extrator : HOG",
            "extrator_orientacoes: " + str(confExt[1]),
            "extrator_pixel_por_celula : " + str(confExt[0]),
            "extrator_celula_por_bloco : " + str(confExt[2]) + "\n",
        ]

    elif extrator.startswith("LBP", 0, 3):
        configText = [
            "extrator : LBP",
            "extrator_numero_pontos: " + str(confExt[0]),
            "extrator_raio : " + str(confExt[1]),
            "extrator_metodo : uniform\n",
        ]

    configText += [
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

    config.close()

'''
41 = A
42 = B
43 = C
44 = D
45 = E
46 = F
47 = G
48 = H
49 = I
4a = J
4b = K
4c = L
4d = M
4e = N
4f = O
50 = P
51 = Q
52 = R
53 = S
54 = T
55 = U
56 = V
57 = W
58 = X
59 = Y
5a = Z
'''
dic1 = {'_41_': (1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_42_': (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_43_': (0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_44_': (0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_45_': (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_46_': (0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_47_': (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_48_': (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_49_': (0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4a_': (0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4b_': (0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4c_': (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4d_': (0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4e_': (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0),
        '_4f_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
        '_50_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0),
        '_51_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0),
        '_52_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0),
        '_53_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0),
        '_54_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0),
        '_55_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0),
        '_56_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0),
        '_57_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0),
        '_58_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0),
        '_59_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0),
        '_5a_': (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)}
dic2 = {(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'A',
        (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'B',
        (0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'C',
        (0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'D',
        (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'E',
        (0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'F',
        (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'G',
        (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'H',
        (0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'I',
        (0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'J',
        (0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'K',
        (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'L',
        (0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0) : 'M',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0) : 'N',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0) : 'O',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0) : 'P',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0) : 'Q',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0) : 'R',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0) : 'S',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0) : 'T',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0) : 'U',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0) : 'V',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0) : 'W',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0) : 'X',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0) : 'Y',
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1) : 'Z'}

    try:
        arquivosPasta = os.listdir(caminhoEntrada)
    except OSError as err:
        print("Erro ao listar arquivos da pasta de entrada: ",err)
        return

    arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

    if len(arquivosImagem) == 0:
        print("Pasta selecionada nao contem imagens .txt!")
        return

    ordenados = sorted(arquivosImagem)

    listaS = ordenados[0:1000]
    listaX = ordenados[1000:2000]
    listaZ = ordenados[2000:3000]

    fold1 = listaS[0:200] + listaX[0:200] + listaZ[0:200]
    fold2 = listaS[200:400] + listaX[200:400] + listaZ[200:400]
    fold3 = listaS[400:600] + listaX[400:600] + listaZ[400:600]
    fold4 = listaS[600:800] + listaX[600:800] + listaZ[600:800]
    fold5 = listaS[800:1000] + listaX[800:1000] + listaZ[800:1000]

    listaFolds = [fold1, fold2, fold3, fold4, fold5]

    pesosV = None
    pesosW = None
    for i in range(5):
        teste = listaFolds[i]
        treinamento = []
        for j in range(5):
            if(j != i):
                treinamento = treinamento + listaFolds[j]

        shuffle(treinamento) # embaralha a ordem dos arquivos de treinamento

        somaTotalErro = 0    
        for arquivo in treinamento:     # loop de treinamento dos folds atuais
                
            entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
            entrada = np.append(entrada, [1.])
            entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna
            letra = None
            if "_53_" in arquivo:
                saidaEsperada = np.asarray(dic1['_53_'])
                letra = "S"
            elif "_58_" in arquivo:
                saidaEsperada = np.asarray(dic1['_58_'])
                letra = "X"
            elif "_5a_" in arquivo:
                saidaEsperada = np.asarray(dic1['_5a_'])
                letra = "Z"
            else:
                print("\nERRO: arquivo de entrada nao eh 'S', nem 'X' nem 'Z'! (nome errado ou alterado)\n")

            saida, pesosV, pesosW, erroTreinamento = MLP(entrada,alfa,epocas,erroMaximo,nroNeuronios,saidaEsperada, pesosV, pesosW)

            somaTotalErro = somaTotalErro + erroTreinamento
            '''
            saidaArredondada = []
            for aux in range(saida.size):
                saidaArredondada.append(round(saida[0][aux], 0))
            try:
                letraFinal = str(dic2[tuple(saidaArredondada)])
            except KeyError:
                letraFinal = None
            '''
        erroFinal = somaTotalErro / len(treinamento)
        erro.write(str(i) + ";" + str(erroFinal))

        somaTotalErroTeste = 0
        for arquivo in teste:       # loop de teste dos folds
            entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
            entrada = np.append(entrada, [1.])
            entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

            saidaTestada = testaMLP(entrada, pesosV, pesosW)
            erroQuadraticoTeste = erroQuadratico(saidaTestada, saidaEsperada)
            somaTotalErroTeste = somaTotalErroTeste + erroQuadraticoTeste
        
        erroFinalTeste = somaTotalErroTeste / len(teste)
        erro.write(";" + str(erroFinalTeste) + "\n")

    data = (pesosV, pesosW)
    pickle.dump(data, model)

    model.close()
    erro.close()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
    print("Fim da execucao da rede em " + str(st) + "\n\n")

if __name__ == "__main__":
    main(sys.argv)