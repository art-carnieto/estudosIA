#coding: utf-8
# script para passar todas as imagens no filtro
# HOG e escreve-las em uma pasta na saida

'''
53 = S ==> 200 em cada k fold
58 = X ==> 200 em cada k fold
5a = Z ==> 200 em cada k fold
'''

import numpy as np
import os

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo hog.py
	caminhoEntrada = "/home/ubuntu/Downloads/dataset1/treinamento/HOG 32" # pasta selecionada pelo usuario
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.txt' in k, arquivosPasta))

ordenados = sorted(arquivosImagem)

arq = open("lista de arquivos.txt", "w")

for i in arquivosImagem:
	arq.write(i + "\n")

arq.close()

listaS = ordenados[0:1000]
listaX = ordenados[1000:2000]
listaZ = ordenados[2000:3000]

'''
print(len(listaS))
print(len(listaX))
print(len(listaZ))
'''

fold1 = listaS[0:200] + listaX[0:200] + listaZ[0:200]
fold2 = listaS[200:400] + listaX[200:400] + listaZ[200:400]
fold3 = listaS[400:600] + listaX[400:600] + listaZ[400:600]
fold4 = listaS[600:800] + listaX[600:800] + listaZ[600:800]
fold5 = listaS[800:1000] + listaX[800:1000] + listaZ[800:1000]

listaFolds = [fold1, fold2, fold3, fold4, fold5]
exemplo = [0, 1, 2, 3, 4]

arq = open("OPEN.txt", "w")


for i in range(5):
	treinamento = []
	arq.write("\ni = " + str(i) + "\n\n")
	teste = listaFolds[i]
	for j in range(5):
		arq.write("\nj = " + str(j) + "\n\n")
		if(j != i):
			arq.write("\n=============================================================\n")
			treinamento = treinamento + listaFolds[j]
	arq.write("\n*****************TESTES*******************\n")	
	arq.write(str(teste))
	arq.write("\n*****************TREINAMENTO*******************\n")
	arq.write(str(treinamento))
	# teste = lista de strings com os arquivos para testar depois de rodar o treinamento
	# treinamento = lista de strings para rodar o treinamento da rede
	# depois desses dois for ja pode usar o k fold na rede!