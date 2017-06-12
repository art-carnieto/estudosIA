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
    return np.random.random((linhas,colunas)) - 1

def MLP(dadosTreinamento, dadosTeste,taxaDeAprendizado,epocas,erroMaximo,nroNeuronios, pesosV, pesosW):
    
    todosErrosEpocas = []
    todosErrosValidacao = []
    
    #TESTE TREINAMENTO
    for epoca in range(epocas):
        erroTreinamento = np.zeros((0,26))
        erroValidacao = np.zeros((0,26))

        #VERIFICAR SE FUNCIONAM ESSES SHAPES!!!!!!!!!!!!!!!!
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
        
        #shuffle(dados) # embaralha a ordem dos arquivos de treinamento e teste ao mesmo tempo

    todosErrosEpocas = np.asarray(todosErrosEpocas)
    todosErrosValidacao = np.asarray(todosErrosValidacao)
     
    return Y, v, w, todosErrosEpocas, todosErrosValidacao

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
    global dic1
    global dic2
    
    extrator = str(argv[1])
    alfa = float(argv[2])
    epocas = int(argv[3])
    erroMaximo = float(argv[4])
    nroNeuronios = int(argv[5])

    caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/" # pasta selecionada pelo usuario
    #/IA/dataset1/HOG1
    #/home/arthur/SI/IA/EP/dataset1/treinamento/
    #C:\\Users\\MICRO 3\\Desktop\\arthur\\dataset1\\treinamento
    
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

    #matrizConfusao = np.zeros((26,26))

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

    matrixTodasImagens = np.zeros((0,4))
    
    for arquivo in ordenados: 

        entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
        entrada = np.append(entrada, [1.])
        if len(entrada.shape) == 1:
            entrada.shape = (entrada.shape[0], 1)
        entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna
       
        #if not 'matrixTodasImagens' in locals():
            #matrixTodasImagens = np.zeros((0,4))

        saida,letra,indice = geraSaidaEsperada(arquivo)

        auxMeta = np.array([saida,letra,indice,entrada])
        
        matrixTodasImagens = np.vstack([matrixTodasImagens,auxMeta])

    #imagensPorLetra = matrixTodasImagens.shape[0] / 26

    #listaLetras = np.zeros((26,imagensPorLetra), dtype='object')
    
    #dados[linha][informação (saida,letra,indice,entrada)][segunda dimenção da informação (para saida / entrada)]
    
    #dados[linha][3][iterador] -> entrada (X)
    #dados[linha][0][iterador] -> saida (target)
    
    '''
    for i in range(listaLetras.shape[0]): #letras
        for j in range(listaLetras.shape[1]): #imagem em letras
            listaLetras[i][j] = matrixTodasImagens[i * imagensPorLetra + j]
            print(listaLetras[i][j][1])
    
    for i in range(3): #letras
        print(listaLetras[9][i][1])
        
    print(listaLetras.shape[0])
    print(listaLetras.shape[1])
    print("-------")
    print(listaLetras[9][1][1])
    
    return
    '''

    np.set_printoptions(threshold = np.nan)

    logTeste = open('testePrints.txt', 'w')

    # logTeste.write("matrixTodasImagens" + '     len = ' + str(len(matrixTodasImagens)) + '     shape = ' + str(matrixTodasImagens.shape) + "\n")
    # logTeste.write(str(matrixTodasImagens) + '\n\n')

    # logTeste.write("matrixTodasImagens[0]" + '     len = ' + str(len(matrixTodasImagens[0])) + '     shape = ' + str(matrixTodasImagens[0].shape) + "\n")
    # logTeste.write(str(matrixTodasImagens[0]) + '\n\n')

    # logTeste.write("matrixTodasImagens[0][0]" + '     len = ' + str(len(matrixTodasImagens[0][0])) + '     shape = ' + str(matrixTodasImagens[0][0].shape) + "\n")
    # logTeste.write(str(matrixTodasImagens[0][0]) + '\n\n')
    # logTeste.write("matrixTodasImagens[0][1]" + '     len = ' + str(len(matrixTodasImagens[0][1])) + '     shape = Não tem shape!\n')
    # logTeste.write(str(matrixTodasImagens[0][1]) + '\n\n')
    # logTeste.write("matrixTodasImagens[0][2]" + '     len = Não tem len!     shape = Não tem shape!\n')
    # logTeste.write(str(matrixTodasImagens[0][2]) + '\n\n')
    # logTeste.write("matrixTodasImagens[0][3]" + '     len = ' + str(len(matrixTodasImagens[0][3])) + '     shape = ' + str(matrixTodasImagens[0][3].shape) + "\n")
    # logTeste.write(str(matrixTodasImagens[0][3]) + '\n\n')

    # logTeste.write("matrixTodasImagens[2999]" + '     len = ' + str(len(matrixTodasImagens[2999])) + '     shape = ' + str(matrixTodasImagens[2999].shape) + "\n")
    # logTeste.write(str(matrixTodasImagens[2999]) + '\n\n')

    # logTeste.write("\n\n\n")

    # logTeste.write("matrixTodasImagens[0:1000]" + '     len = ' + str(len(matrixTodasImagens[0:1000])) + '     shape = ' + str(matrixTodasImagens[0:1000].shape) + "\n\n")

    # for i in range(matrixTodasImagens[0:1000].shape[0]):
    #     logTeste.write(str(matrixTodasImagens[0:1000][i][1]) + '\n')

    # logTeste.write("\n\n\n")

    # logTeste.write("matrixTodasImagens[1000:2000]" + '     len = ' + str(len(matrixTodasImagens[1000:2000])) + '     shape = ' + str(matrixTodasImagens[1000:2000].shape) + "\n\n")

    # for i in range(matrixTodasImagens[1000:2000].shape[0]):
    #     logTeste.write(str(matrixTodasImagens[1000:2000][i][1]) + '\n')

    # logTeste.write("\n\n\n")

    # logTeste.write("matrixTodasImagens[2000:3000]" + '     len = ' + str(len(matrixTodasImagens[2000:3000])) + '     shape = ' + str(matrixTodasImagens[2000:3000].shape) + "\n\n")

    # for i in range(matrixTodasImagens[2000:3000].shape[0]):
    #     logTeste.write(str(matrixTodasImagens[2000:3000][i][1]) + '\n')

    # logTeste.write("\n\n\n")


    listaS = matrixTodasImagens[0:1000]
    listaX = matrixTodasImagens[1000:2000]
    listaZ = matrixTodasImagens[2000:3000]
    
    fold1 = np.vstack((listaS[0:200], listaX[0:200], listaZ[0:200]))
    fold2 = np.vstack((listaS[200:400], listaX[200:400], listaZ[200:400]))
    fold3 = np.vstack((listaS[400:600], listaX[400:600], listaZ[400:600]))
    fold4 = np.vstack((listaS[600:800], listaX[600:800], listaZ[600:800]))
    fold5 = np.vstack((listaS[800:1000], listaX[800:1000], listaZ[800:1000]))

    # logTeste.write("fold1" + '     len = ' + str(len(fold1)) + '     shape = ' + str(fold1.shape) + "\n")
    # logTeste.write("fold2" + '     len = ' + str(len(fold2)) + '     shape = ' + str(fold2.shape) + "\n")
    # logTeste.write("fold3" + '     len = ' + str(len(fold3)) + '     shape = ' + str(fold3.shape) + "\n")
    # logTeste.write("fold4" + '     len = ' + str(len(fold4)) + '     shape = ' + str(fold4.shape) + "\n")
    # logTeste.write("fold5" + '     len = ' + str(len(fold5)) + '     shape = ' + str(fold5.shape) + "\n\n")

    # logTeste.write("fold1:\n")
    # for i in range(fold1.shape[0]):
    #     logTeste.write(str(fold1[i][1]) + '\n')

    # logTeste.write("fold2:\n")
    # for i in range(fold2.shape[0]):
    #     logTeste.write(str(fold2[i][1]) + '\n')

    # logTeste.write("fold3:\n")
    # for i in range(fold3.shape[0]):
    #     logTeste.write(str(fold3[i][1]) + '\n')

    # logTeste.write("fold4:\n")
    # for i in range(fold4.shape[0]):
    #     logTeste.write(str(fold4[i][1]) + '\n')

    # logTeste.write("fold5:\n")
    # for i in range(fold5.shape[0]):
    #     logTeste.write(str(fold5[i][1]) + '\n')

    # logTeste.write("\n\n\n")

    listaFolds = [fold1, fold2, fold3, fold4, fold5]
    
    # logTeste.write("listaFolds" + '     len = ' + str(len(listaFolds)) + '     shape = Não tem shape!\n')

    # logTeste.write("listaFolds[0]" + '     len = ' + str(len(listaFolds[0])) + '     shape = ' + str(listaFolds[0].shape) + '\n')
    # logTeste.write("listaFolds[1]" + '     len = ' + str(len(listaFolds[1])) + '     shape = ' + str(listaFolds[1].shape) + '\n')
    # logTeste.write("listaFolds[2]" + '     len = ' + str(len(listaFolds[2])) + '     shape = ' + str(listaFolds[2].shape) + '\n')
    # logTeste.write("listaFolds[3]" + '     len = ' + str(len(listaFolds[3])) + '     shape = ' + str(listaFolds[3].shape) + '\n')
    # logTeste.write("listaFolds[4]" + '     len = ' + str(len(listaFolds[4])) + '     shape = ' + str(listaFolds[4].shape) + '\n')

    # logTeste.write("\n\n\n")

    # for fold in range(len(listaFolds)):
    #     for i in range(listaFolds[fold].shape[0]):
    #         logTeste.write('[' + str(i) + '] ==> ' + str(listaFolds[fold][i][1]) + '\n')
    #     logTeste.write('\n\n')

    # for i in range(listaFolds[0].shape[0]):
    #     logTeste.write('[' + str(i) + '] ==> ' + str(listaFolds[0][i][1]) + ' '
    #                                         + str(listaFolds[1][i][1]) + ' '
    #                                         + str(listaFolds[2][i][1]) + ' '
    #                                         + str(listaFolds[3][i][1]) + ' '
    #                                         + str(listaFolds[4][i][1]) + ' ' + '\n')

    vInicial = criaMatrizPesosDefault(matrixTodasImagens[0][3].shape[1],nroNeuronios)
    wInicial = criaMatrizPesosDefault(nroNeuronios,matrixTodasImagens[0][0].shape[0])
    
    #folds
    for i in range(len(listaFolds)):
        logTeste.write('i = ' + str(i) + '\n')
        teste = listaFolds[i]
        treinamento = np.zeros((0, 4))
        for j in range(len(listaFolds)):
            if(j != i):
                logTeste.write('   j = ' + str(j) + '\n')
                treinamento = np.vstack((treinamento, listaFolds[j]))

        logTeste.write("teste" + '     len = ' + str(len(teste)) + '     shape = ' + str(teste.shape) + '\n')
        logTeste.write("treinamento" + '     len = ' + str(len(treinamento)) + '     shape = ' + str(teste.shape) + '\n')

        logTeste.write("treinamento[0]" + '     len = ' + str(len(treinamento[0])) + '     shape = ' + str(treinamento[0].shape) + "\n")
        logTeste.write(str(treinamento[0]) + '\n\n')
        logTeste.write("treinamento[599]" + '     len = ' + str(len(treinamento[599])) + '     shape = ' + str(treinamento[599].shape) + "\n")
        logTeste.write(str(treinamento[599]) + '\n\n')
        logTeste.write("treinamento[600]" + '     len = ' + str(len(treinamento[600])) + '     shape = ' + str(treinamento[600].shape) + "\n")
        logTeste.write(str(treinamento[600]) + '\n\n')
        logTeste.write("treinamento[2399]" + '     len = ' + str(len(treinamento[2399])) + '     shape = ' + str(treinamento[2399].shape) + "\n")
        logTeste.write(str(treinamento[2399]) + '\n\n')

        saida, pesosV, pesosW, errosTreino, errosValidacao = MLP(treinamento,teste,alfa,epocas,erroMaximo,nroNeuronios, vInicial, wInicial)

        plt.plot(errosTreino)
        plt.plot(errosValidacao)
        plt.title('Rodada ' + str(i))
        plt.ylabel('Erro')
        plt.xlabel('Epoca')
        nomePlot = 'erros_rodada' + str(i) + '.png'
        plt.savefig(os.path.join(caminhoSaida, nomePlot))
        plt.close()

        for k in range(len(errosTreino)-1):
            erro.write(str(k) + ';' + str(errosTreino[k]) + ';' + str(errosValidacao[k]) + '\n')
        erro.write(str(k+1) + ';' + str(errosTreino[len(errosTreino)-1]) + ';' + str(errosValidacao[len(errosTreino)-1]) + '\n\n')

        somaTotalErroTeste = 0
        for arquivo in teste:       # loop de teste dos folds
            entrada = np.loadtxt(os.path.join(caminhoEntrada, arquivo), dtype='float', delimiter="\n")
            entrada = np.append(entrada, [1.])
            entrada = np.transpose(entrada) # transpoe de uma matriz linha para uma matriz coluna

            saidaTestada = testaMLP(entrada, pesosV, pesosW)
            saidaEsperadaTeste, letraTeste, indiceTeste = geraSaidaEsperada(arquivo)

            erroQuadraticoTeste = erroQuadratico(saidaTestada, saidaEsperadaTeste)
            somaTotalErroTeste = somaTotalErroTeste + erroQuadraticoTeste

            saidaArredondadaTeste = []
            for aux in range(saida.size):
                saidaArredondadaTeste.append(round(saidaTestada[0][aux], 0))

            somaColuna(matrizConfusao, saidaArredondadaTeste, indiceTeste)
        
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