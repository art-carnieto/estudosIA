#coding: utf-8
# codigo da rede neural MLP

import numpy as np
from scipy.misc import derivative
import math

import os

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def derSigmoid(x):
	return math.exp(x) / math.pow((math.exp(x)+1),2)

"""

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo rede.py
	caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento/HOG 32" # pasta selecionada pelo usuario
	arquivosDaPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivos = list(filter(lambda k: '.txt' in k, arquivosDaPasta))

if len(arquivos) == 0:
	print("Pasta selecionada nao contem imagens .txt")

print("Iniciando rede neural LBP")
for arquivo in arquivos:
	print("\tProcessando arquivo " + arquivo)
	
	entrada = np.loadtxt(os.path.join(caminhoEntrada,arquivo), dtype='float', delimiter="\n")
	print("Neuronios = ")
	print(entrada)
	
"""

"""
	try:
		caminhoSaida = os.path.join(caminhoEntrada,"HOG " + str(i))
		if not os.path.exists(caminhoSaida):
			os.makedirs(caminhoSaida)
	except OSError as err:
		print("Erro de acesso a pasta de saida: ", err)

	try:
		f = open(os.path.join(caminhoSaida, saida), 'w')
		# f.write(str(v)[2:-1])	# tem esse corte [2:-1] para tirar '[' e ']' da saida
		for x in np.nditer(v):
			f.write(str(x) + "\n")
	except IOError as err:
		print("Erro na escrita do arquivo ", saida, ": ", err)



arquivo = "/home/arthur/SI/IA/EP/dataset1/treinamento/HOG 32/train_5a_00000.txt"

print("\tProcessando arquivo " + arquivo)
	
entrada = np.loadtxt(arquivo, dtype='float', delimiter="\n")
print("Neuronios = ")
print(entrada)

"""

#ultima coluna = bias
entrada = np.array( [  [0., 0., 1.],
					   [0., 1., 1.], 
					   [1., 0., 1.], 
					   [1., 1., 1.] ])

nroNeuronios = 5	#definido pelo usuario

np.random.seed(1)

#ultima linha = bias
pesos1 = np.random.rand(3,nroNeuronios)

"""
pesos1 = np.array ([ [, , , , ],
					 [, , , , ],
					 [, , , , ]])
"""

#ultima linha = bias
pesos2 = np.random.rand(nroNeuronios+1,1)

"""
pesos2 = np.random.rand( [ [],
						   [],
						   [],
						   [],
						   [],
						   []])
"""
# saidaEsperada = np.array( [0., 1., 1., 0.] )

saidaEsperada = np.array( [ [0.],
							[1.],
							[1.],
							[0.]] )

taxaAprendizado = 0.5

nroEpocas = 1

print("")
print("Entrada (X) = ")
print(entrada)

print("")
print("pesos1 (v) = ")
print(pesos1)

print("")
print("taxaAprendizado (α)= " + str(taxaAprendizado))
print("nroEpocas = " + str(nroEpocas))

for epoca in range(nroEpocas):
	print("")
	print("***Epoca = " + str(epoca))



	###########FEED FORWARD###########


	escondida = np.dot(entrada, pesos1)
	print("escondida depois da multiplicacao (Z_in):")
	print(escondida)

	escondidaFx = escondida.copy()

	print("")
	print("FUNCAO ATIVACAO CAMADA ESCONDIDA:")
	j = 0
	for i in np.nditer(escondidaFx):
		print("   i = " + str(i) + "\tj = " + str(j) + "\tsig = " + str(sigmoid(i)))
		escondidaFx.itemset(j, sigmoid(i))
		j += 1
	print("************************")

	print("")
	print("escondida depois de ativar (Z):")
	print(escondidaFx)

	# adicionando uma nova coluna no final da camada escondida = bias
	aux = np.zeros((escondida.shape[0], escondida.shape[1]+1))
	aux[:,:-1] = escondidaFx
	aux[:,-1:] = 1.
	escondidaFx = aux
	
	print("")
	print("escondida depois de adicionar o bias (Z) :")
	print(escondidaFx)

	print("")
	print("pesos2 (w) = ")
	print(pesos2)

	saida = np.dot(escondidaFx, pesos2)
	print("")
	print("saida (Y_in) = ")
	print(saida)
	
	saidaAtivada = saida.copy()

	print("")
	print("FUNCAO ATIVACAO SAIDA:")

	'''
	j = 0
	for i in np.nditer(saidaAtivada):
		print("   i = " + str(i) + "\tj = " + str(j) + "\tsig = " + str(sigmoid(i)))
		saidaAtivada.itemset(j, sigmoid(i))
		j += 1
	'''

	for i in range(saidaAtivada.size):
		saidaAtivada[i] = sigmoid(saidaAtivada[i])

	print("************************")

	print("")
	print("saida ativada (Y) :")
	print(saidaAtivada)


	###########CALCULO DO ERRO###########

	calcErro = saidaEsperada - saidaAtivada

	print("")
	print("calcErro antes de derivar (δ = T - Y) : ")
	print(calcErro)

	print("")
	print("saida esperada")
	print(saidaEsperada)

	print("")
	print("calcErro FINAL: (δ = (T - Y)*sig'(Y_in)):")
	
	for i in range(calcErro.size):
		calcErro[i] = calcErro[i] * derSigmoid(saida[i])

	print(calcErro)

	###########BACK PROPAGATION###########

	escondidaTransposta = np.transpose(escondidaFx.copy())	
	multiDePropagacao = np.dot(escondidaTransposta,calcErro)
	deltaW = taxaAprendizado * multiDePropagacao

	print("--------------")
	print(pesos2)

	pesos2 = pesos2 + deltaW
	
	print("--------------")
	print(pesos2)




	'''
	deltaW = np.dot(calcErro, pesos2)
	print("")
	print("deltaW:")
	print(deltaW)
	'''