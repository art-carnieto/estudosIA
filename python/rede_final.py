#coding: utf-8
import numpy as np
import os
from random import shuffle
import time
import datetime
from random import shuffle
import pickle
import sys
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 500

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

# np.random.seed(1) #semente de aleatoriedade

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derSigmoid(x):
    return np.exp(x) / np.power((np.exp(x)+1),2)

def erroQuadratico(erros):
    return np.sum(np.power(erros,2))/(2*len(erros))

def criaMatrizPesosDefault(linhas,colunas):
    return np.random.random((linhas,colunas))

def modificaTaxaAprendizado(taxaAprendizado):
    return (6*taxaAprendizado)/7 
    

def MLP(dadosTreinamento,dadosTeste,taxaDeAprendizado,epocas,erroMaximo,nroNeuronios,pesosV,pesosW):
    
    todosErrosEpocas = []
    todosErrosValidacao = []
    
    #TESTE TREINAMENTO
    for epoca in range(epocas):
        erroTreinamento = np.zeros((0,26))
        erroValidacao = np.zeros((0,26))

        modificaTaxaAprendizado(taxaDeAprendizado)

        for i in range(dadosTreinamento.shape[0]):     # loop de treinamento dos folds atuais
            
            X = dadosTreinamento[i][3]
            if len(X.shape) == 1:
                X.shape = (X.shape[0], 1)

            X = np.transpose(X)

            T = dadosTreinamento[i][0]
            
            if len(T.shape) == 1:
                T.shape = (T.shape[0], 1)
            
            X = np.transpose(X)
            T = np.transpose(T)
            
            v = pesosV
            w = pesosW
            
            #calculo dos valores e ativação
            Z_in = np.dot(X,v)

            Z = sigmoid(Z_in)

            Y_in = np.dot(Z,w)

            Y = sigmoid(Y_in)

            #erro (diferença entre o target)
            taxaDeErroSaida = T - Y
            erroTreinamento = np.vstack([erroTreinamento, taxaDeErroSaida])
                
            #taxa de erro para segunda camada de pesos (δw[k])
            taxaDeErroW = taxaDeErroSaida * derSigmoid(Y_in)
            
            #∆w[j][k] = α*δw[k]*Z[j]
            deltaW = taxaDeAprendizado * np.dot(np.transpose(Z),taxaDeErroW)

            #erro para V (δv_inv[j] = ∑ k=1 δw[k]w[j][k] )
            taxaDeErroEscondida = taxaDeErroW.dot(np.transpose(w))

            # δv[j] = δv_in[j] f′(z_in[j])
            taxaDeErroV = taxaDeErroEscondida * derSigmoid(Z_in)

            #∆v[i][j] = αδv[j]x[i]
            deltaV = taxaDeAprendizado * np.dot(np.transpose(X),taxaDeErroV)

            #w[j][k](new) = w[j][k](old) + ∆w[j][k]
            w += deltaW

            #v[i][j](new) = v[i][j](old) + ∆v[i][j]
            v += deltaV
        
        #TESTE VALIDACAO
        for i in range(dadosTeste.shape[0]):     # loop de teste do fold de teste atual
            X = dadosTeste[i][3]
            T = dadosTeste[i][0]
            Z_in = np.dot(X,pesosV)
            Z = sigmoid(Z_in)
            Y_in = np.dot(Z,pesosW)
            Y = sigmoid(Y_in)
            erroValidacao = np.vstack([erroValidacao, (T - Y)])

        erroValidacao = np.asarray(erroValidacao)

        erroTreinoEpoca = erroQuadratico(erroTreinamento)
        erroValidacaoEpoca = erroQuadratico(erroValidacao)

        todosErrosEpocas.append(erroTreinoEpoca)
        todosErrosValidacao.append(erroValidacaoEpoca)
        
        if(erroTreinoEpoca < erroMaximo): 
            print ("erro a baixo do maximo")
            return v, w, todosErrosEpocas, todosErrosValidacao
        
        # if epoca > 0:
        #     if todosErrosValidacao[epoca] > todosErrosValidacao[epoca-1]:
        #         print("SAIU PORQUE ERRO DE VALIDAÇÃO AUMENTOU")
        #         print(todosErrosValidacao[epoca])
        #         print(todosErrosValidacao[epoca-1])

        #         todosErrosEpocas = np.asarray(todosErrosEpocas)
        #         todosErrosValidacao = np.asarray(todosErrosValidacao)
                 
        #         return v, w, todosErrosEpocas, todosErrosValidacao
        
        np.random.shuffle(dadosTreinamento) # embaralha a ordem dos folds de treinamento
        np.random.shuffle(dadosTeste) # embaralha a ordem dos fold de treino

    todosErrosEpocas = np.asarray(todosErrosEpocas)
    todosErrosValidacao = np.asarray(todosErrosValidacao)
     
    return v, w, todosErrosEpocas, todosErrosValidacao

def testaMLP(entrada, pesosV, pesosW):
    if len(entrada.shape) == 1:
        entrada.shape = (entrada.shape[0],1)

    X = np.copy(np.transpose(entrada.copy()))

    Z_in = np.dot(X,pesosV)

    Z = sigmoid(Z_in)

    Y_in = np.dot(Z,pesosW)

    Y = sigmoid(Y_in)

    return Y

def geraSaidaEsperada(arquivo):
    if "_41_" in arquivo:
        saidaEsperada = np.asarray(dic1['_41_'])
        letra = "A"
        indice = 0
    elif "_42_" in arquivo:
        saidaEsperada = np.asarray(dic1['_42_'])
        letra = "B"
        indice = 1
    elif "_43_" in arquivo:
        saidaEsperada = np.asarray(dic1['_43_'])
        letra = "C"
        indice = 2
    elif "_44_" in arquivo:
        saidaEsperada = np.asarray(dic1['_44_'])
        letra = "D"
        indice = 3
    elif "_45_" in arquivo:
        saidaEsperada = np.asarray(dic1['_45_'])
        letra = "E"
        indice = 4
    elif "_46_" in arquivo:
        saidaEsperada = np.asarray(dic1['_46_'])
        letra = "F"
        indice = 5
    elif "_47_" in arquivo:
        saidaEsperada = np.asarray(dic1['_47_'])
        letra = "G"
        indice = 6
    elif "_48_" in arquivo:
        saidaEsperada = np.asarray(dic1['_48_'])
        letra = "H"
        indice = 7
    elif "_49_" in arquivo:
        saidaEsperada = np.asarray(dic1['_49_'])
        letra = "I"
        indice = 8
    elif "_4a_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4a_'])
        letra = "J"
        indice = 9
    elif "_4b_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4b_'])
        letra = "K"
        indice = 10
    elif "_4c_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4c_'])
        letra = "L"
        indice = 11
    elif "_4d_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4d_'])
        letra = "M"
        indice = 12
    elif "_4e_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4e_'])
        letra = "N"
        indice = 13
    elif "_4f_" in arquivo:
        saidaEsperada = np.asarray(dic1['_4f_'])
        letra = "O"
        indice = 14
    elif "_50_" in arquivo:
        saidaEsperada = np.asarray(dic1['_50_'])
        letra = "P"
        indice = 15
    elif "_51_" in arquivo:
        saidaEsperada = np.asarray(dic1['_51_'])
        letra = "Q"
        indice = 16
    elif "_52_" in arquivo:
        saidaEsperada = np.asarray(dic1['_52_'])
        letra = "R"
        indice = 17
    elif "_53_" in arquivo:
        saidaEsperada = np.asarray(dic1['_53_'])
        letra = "S"
        indice = 18
    elif "_54_" in arquivo:
        saidaEsperada = np.asarray(dic1['_54_'])
        letra = "T"
        indice = 19
    elif "_55_" in arquivo:
        saidaEsperada = np.asarray(dic1['_55_'])
        letra = "U"
        indice = 20
    elif "_56_" in arquivo:
        saidaEsperada = np.asarray(dic1['_56_'])
        letra = "V"
        indice = 21
    elif "_57_" in arquivo:
        saidaEsperada = np.asarray(dic1['_57_'])
        letra = "W"
        indice = 22
    elif "_58_" in arquivo:
        saidaEsperada = np.asarray(dic1['_58_'])
        letra = "X"
        indice = 23
    elif "_59_" in arquivo:
        saidaEsperada = np.asarray(dic1['_59_'])
        letra = "Y"
        indice = 24
    elif "_5a_" in arquivo:
        saidaEsperada = np.asarray(dic1['_5a_'])
        letra = "Z"
        indice = 25
    else:
        print("\nERRO: arquivo de entrada nao faz parte do conjunto de letras! (nome errado ou alterado)\n")
        return None, None, None

    return saidaEsperada, letra, indice

def somaColuna(matrizBase, vetorSoma, col):
    for i in range(len(matrizBase)):
        matrizBase[i][col] = matrizBase[i][col] + vetorSoma[i]

    return matrizBase

def main(argv):

    if(len(argv) < 7):
        print("Numero errado de argumentos!")
        print("Usagem do rede_final.py:")
        print("argumento-01: Dataset utilizado (número 1, 2 ou 3)")
        print("argumento-02: Pasta com a entrada da rede, deve comecar por 'HOG' ou 'LBP' (sem aspas)")
        print("argumento-03: Alfa (taxa de aprendizado) a ser usado na rede")
        print("argumento-04: Numero de epocas que sera usado no MLP")
        print("argumento-05: Erro maximo da rede MLP")
        print("argumento-06: Numero de neuronios da camada escondida da rede MLP")
        return

    if not(argv[2].startswith("HOG", 0, 3) or argv[2].startswith("LBP", 0, 3)):
        print("Extrator desconhecido!")
        print("O argumento-01 deve ser o nome da pasta com a entrada e deve comecar por 'HOG' ou 'LBP' (sem aspas), por exemplo: 'HOG', 'HOG1', 'HOG2', ...")
        return
    
    global dic1
    global dic2
    
    escolhaDataset = int(argv[1])
    extrator = str(argv[2])
    alfa = float(argv[3])
    epocas = int(argv[4])
    erroMaximo = float(argv[5])
    nroNeuronios = int(argv[6])
    letras = 0

    #pastaBase = "/home/arthur/SI/IA/EP/" # pasta selecionada pelo usuario
    pastaBase = "../../" 
    #pastaBase = "/IA/dataset1/HOG1"
    #pastaBase = "/home/arthur/SI/IA/EP/"
    #pastaBase = "C:\\Users\\MICRO 2\\Desktop\\arthur"
    #pastaBase = "C:\\Users\\MICRO 3\\Desktop\\arthur"
    #pastaBase = "C:\\Users\\Arthur\\Dropbox\\SI\\IA"
    
    if escolhaDataset == 1:
        pastaBase = os.path.join(pastaBase, "dataset1")
        letras = 3
    elif escolhaDataset == 2:
        pastaBase = os.path.join(pastaBase, "dataset2")
        letras = 26
    elif escolhaDataset == 3:
        pastaBase = os.path.join(pastaBase, "datasetRec")
        letras = 10
    else:
        print("Dataset escolhido invalido! Deve ser o número 1, 2 ou 3.")
        return

    try:
        caminhoEntradaTreino = os.path.join(pastaBase, "treinamento", extrator)
        arqExtrator = open(os.path.join(caminhoEntradaTreino, "configExtrator.dat"), "rb")
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
        "rede_camada_Y_neuronios : " + str(letras),
        "rede_camada_Y_funcao_de_ativacao : sigmoide",
        "rede_inicializacao_pesos : aleatoria",
        "rede_min_epocas : 0",
        "rede_max_epocas : " + str(epocas),
        "rede_parada_antecipada : loop que vai de 0 a max epocas"
        ]

    for i in configText:
        config.write(i + "\n")

    config.close()

    try:
        arquivosPasta = os.listdir(caminhoEntradaTreino)
    except OSError as err:
        print("Erro ao listar arquivos da pasta de entrada: ",err)
        return

    arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

    if len(arquivosImagem) == 0:
        print("Pasta selecionada nao contem imagens .txt!")
        return
    
    ordenados = sorted(arquivosImagem)

    matrixTodasImagens = np.zeros((0,4))

    for arquivo in ordenados: 
        entrada = np.loadtxt(os.path.join(caminhoEntradaTreino, arquivo), dtype='float', delimiter="\n")
        #entrada = np.append(entrada, [1.]) #camada de entrada, falta camada escondida
        if len(entrada.shape) == 1:
            entrada.shape = (entrada.shape[0], 1)
        entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna
        saida,letra,indice = geraSaidaEsperada(arquivo)
        auxMeta = np.array([saida,letra,indice,entrada])
        matrixTodasImagens = np.vstack([matrixTodasImagens,auxMeta])

    if escolhaDataset == 1:

        listaS = matrixTodasImagens[0:1000]
        listaX = matrixTodasImagens[1000:2000]
        listaZ = matrixTodasImagens[2000:3000]
        
        fold1 = np.vstack((listaS[0:200], listaX[0:200], listaZ[0:200]))
        fold2 = np.vstack((listaS[200:400], listaX[200:400], listaZ[200:400]))
        fold3 = np.vstack((listaS[400:600], listaX[400:600], listaZ[400:600]))
        fold4 = np.vstack((listaS[600:800], listaX[600:800], listaZ[600:800]))
        fold5 = np.vstack((listaS[800:1000], listaX[800:1000], listaZ[800:1000]))

    elif escolhaDataset == 2:
        
        listaA = matrixTodasImagens[0:1000]
        listaB = matrixTodasImagens[1000:2000]
        listaC = matrixTodasImagens[2000:3000]
        listaD = matrixTodasImagens[3000:4000]
        listaE = matrixTodasImagens[4000:5000]
        listaF = matrixTodasImagens[5000:6000]
        listaG = matrixTodasImagens[6000:7000]
        listaH = matrixTodasImagens[7000:8000]
        listaI = matrixTodasImagens[8000:9000]
        listaJ = matrixTodasImagens[9000:10000]
        listaK = matrixTodasImagens[10000:11000]
        listaL = matrixTodasImagens[11000:12000]
        listaM = matrixTodasImagens[12000:13000]
        listaN = matrixTodasImagens[13000:14000]
        listaO = matrixTodasImagens[14000:15000]
        listaP = matrixTodasImagens[15000:16000]
        listaQ = matrixTodasImagens[16000:17000]
        listaR = matrixTodasImagens[17000:18000]
        listaS = matrixTodasImagens[18000:19000]
        listaT = matrixTodasImagens[19000:20000]
        listaU = matrixTodasImagens[20000:21000]
        listaV = matrixTodasImagens[21000:22000]
        listaW = matrixTodasImagens[22000:23000]
        listaX = matrixTodasImagens[23000:24000]
        listaY = matrixTodasImagens[24000:25000]
        listaZ = matrixTodasImagens[25000:26000]
        
        fold1 = np.vstack((listaA[0:200], listaB[0:200], listaC[0:200], listaD[0:200],
                           listaE[0:200], listaF[0:200], listaG[0:200], listaH[0:200],
                           listaI[0:200], listaJ[0:200], listaK[0:200], listaL[0:200],
                           listaM[0:200], listaN[0:200], listaO[0:200], listaP[0:200],
                           listaQ[0:200], listaR[0:200], listaS[0:200], listaT[0:200],
                           listaU[0:200], listaV[0:200], listaW[0:200], listaX[0:200],
                           listaY[0:200], listaZ[0:200]))
        fold2 = np.vstack((listaA[200:400], listaB[200:400], listaC[200:400], listaD[200:400],
                           listaE[200:400], listaF[200:400], listaG[200:400], listaH[200:400],
                           listaI[200:400], listaJ[200:400], listaK[200:400], listaL[200:400],
                           listaM[200:400], listaN[200:400], listaO[200:400], listaP[200:400],
                           listaQ[200:400], listaR[200:400], listaS[200:400], listaT[200:400],
                           listaU[200:400], listaV[200:400], listaW[200:400], listaX[200:400],
                           listaY[200:400], listaZ[200:400]))
        fold3 = np.vstack((listaA[400:600], listaB[400:600], listaC[400:600], listaD[400:600],
                           listaE[400:600], listaF[400:600], listaG[400:600], listaH[400:600],
                           listaI[400:600], listaJ[400:600], listaK[400:600], listaL[400:600],
                           listaM[400:600], listaN[400:600], listaO[400:600], listaP[400:600],
                           listaQ[400:600], listaR[400:600], listaS[400:600], listaT[400:600],
                           listaU[400:600], listaV[400:600], listaW[400:600], listaX[400:600],
                           listaY[400:600], listaZ[400:600]))
        fold4 = np.vstack((listaA[600:800], listaB[600:800], listaC[600:800], listaD[600:800],
                           listaE[600:800], listaF[600:800], listaG[600:800], listaH[600:800],
                           listaI[600:800], listaJ[600:800], listaK[600:800], listaL[600:800],
                           listaM[600:800], listaN[600:800], listaO[600:800], listaP[600:800],
                           listaQ[600:800], listaR[600:800], listaS[600:800], listaT[600:800],
                           listaU[600:800], listaV[600:800], listaW[600:800], listaX[600:800],
                           listaY[600:800], listaZ[600:800]))
        fold5 = np.vstack((listaA[800:1000], listaB[800:1000], listaC[800:1000], listaD[800:1000],
                           listaE[800:1000], listaF[800:1000], listaG[800:1000], listaH[800:1000],
                           listaI[800:1000], listaJ[800:1000], listaK[800:1000], listaL[800:1000],
                           listaM[800:1000], listaN[800:1000], listaO[800:1000], listaP[800:1000],
                           listaQ[800:1000], listaR[800:1000], listaS[800:1000], listaT[800:1000],
                           listaU[800:1000], listaV[800:1000], listaW[800:1000], listaX[800:1000],
                           listaY[800:1000], listaZ[800:1000]))

    elif escolhaDataset == 3:
        
        listaA = matrixTodasImagens[0:1000]
        listaB = matrixTodasImagens[1000:2000]
        listaC = matrixTodasImagens[2000:3000]
        listaE = matrixTodasImagens[3000:4000]
        listaH = matrixTodasImagens[4000:5000]
        listaI = matrixTodasImagens[5000:6000]
        listaK = matrixTodasImagens[6000:7000]
        listaM = matrixTodasImagens[7000:8000]
        listaP = matrixTodasImagens[8000:9000]
        listaX = matrixTodasImagens[9000:10000]
        
        fold1 = np.vstack((listaA[0:200], listaB[0:200], listaC[0:200], listaE[0:200], listaH[0:200],
                           listaI[0:200], listaK[0:200], listaM[0:200], listaP[0:200], listaX[0:200]))

        fold2 = np.vstack((listaA[200:400], listaB[200:400], listaC[200:400], listaE[200:400], listaH[200:400],
                           listaI[200:400], listaK[200:400], listaM[200:400], listaP[200:400], listaX[200:400]))

        fold3 = np.vstack((listaA[400:600], listaB[400:600], listaC[400:600], listaE[400:600], listaH[400:600],
                           listaI[400:600], listaK[400:600], listaM[400:600], listaP[400:600], listaX[400:600]))

        fold4 = np.vstack((listaA[600:800], listaB[600:800], listaC[600:800], listaE[600:800], listaH[600:800],
                           listaI[600:800], listaK[600:800], listaM[600:800], listaP[600:800], listaX[600:800]))

        fold5 = np.vstack((listaA[800:1000], listaB[800:1000], listaC[800:1000], listaE[800:1000], listaH[800:1000],
                           listaI[800:1000], listaK[800:1000], listaM[800:1000], listaP[800:1000], listaX[800:1000]))
    
    listaFolds = [fold1, fold2, fold3, fold4, fold5]
    # listaFolds = organizarFolds(matrixTodasImagens, 3, 5, len(arquivosImagem))

    try:
        caminhoEntradaTeste = os.path.join(pastaBase, "testes", extrator)
    except IOError as err:
        print("Erro no acesso a pasta com as imagens de teste da rede.\nDeve ter com o extrator dentro da pasta 'testes' da pasta do dataset com as imagens processadas!",err)
        return

    try:
        arquivosPasta = os.listdir(caminhoEntradaTeste)
    except OSError as err:
        print("Erro ao listar arquivos da pasta de testes: ",err)
        return

    arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

    if len(arquivosImagem) == 0:
        print("Pasta selecionada nao contem imagens .txt!")
        return
    
    ordenados = sorted(arquivosImagem)
    
    matrixImagensTeste = np.zeros((0,4))

    for arquivo in ordenados: 
        entradaTeste = np.loadtxt(os.path.join(caminhoEntradaTeste, arquivo), dtype='float', delimiter="\n") # ## testar com a pasta de testes ou o fold de teste?!
        entradaTeste = np.append(entradaTeste, [1.])
        if len(entradaTeste.shape) == 1:
            entradaTeste.shape = (entradaTeste.shape[0], 1)
        entradaTeste = np.transpose(entradaTeste) # transpoe de uma matriz linha para uma matriz coluna
        saida,letra,indice = geraSaidaEsperada(arquivo)
        auxMeta = np.array([saida,letra,indice,entradaTeste])
        matrixImagensTeste = np.vstack([matrixImagensTeste,auxMeta])

    vInicial = criaMatrizPesosDefault(matrixTodasImagens[0][3].shape[1],nroNeuronios)
    wInicial = criaMatrizPesosDefault(nroNeuronios,matrixTodasImagens[0][0].shape[0])
    
    #folds
    for i in range(len(listaFolds)):
        teste = np.copy(listaFolds[i])
        treinamento = np.zeros((0, 4))
        for j in range(len(listaFolds)):
            if(j != i):
                treinamento = np.vstack((treinamento, np.copy(listaFolds[j])))

        vAux = np.copy(vInicial)
        wAux = np.copy(wInicial)

        pesosV, pesosW, errosTreino, errosValidacao = MLP(treinamento,teste,alfa,epocas,erroMaximo,nroNeuronios, vAux, wAux)

        plt.plot(errosTreino, label='Erro de treinamento')
        plt.plot(errosValidacao, label='Erro de validacao')
        plt.title('Rodada ' + str(i))
        plt.ylabel('Erro')
        plt.xlabel('Epoca')
        nomePlot = 'erros_rodada' + str(i) + '.png'
        plt.legend()
        plt.savefig(os.path.join(caminhoSaida, nomePlot))
        plt.close()

        for k in range(len(errosTreino)-1):
            erro.write(str(k) + ';' + str(errosTreino[k]) + ';' + str(errosValidacao[k]) + '\n')
        erro.write(str(k+1) + ';' + str(errosTreino[len(errosTreino)-1]) + ';' + str(errosValidacao[len(errosTreino)-1]) + '\n\n')

        nomeModel = 'model' + str(i) + '.dat'
        model = open(os.path.join(caminhoSaida,nomeModel), "wb")
        data = (pesosV, pesosW)
        pickle.dump(data, model, protocol=2)
        model.close()

    erro.close
    
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
    print("Fim da execucao da rede em " + str(st) + "\n\n")

if __name__ == "__main__":
    main(sys.argv)