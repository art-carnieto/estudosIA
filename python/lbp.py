#coding: utf-8
# script para passar todas as imagens no filtro
# LBP e escreve-las em uma pasta na saida

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import local_binary_pattern
from skimage import color
from skimage.filters import roberts, sobel
import os

np.set_printoptions(threshold=np.nan) # corrige o tamanho maximo do print na saida de um ndarray

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo lbp.py
	caminhoEntrada = "/home/juuullyanne/Área de Trabalho/IA/dataset1/treinamento" # pasta selecionada pelo usuario
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))

if len(arquivosImagem) == 0:
	print("Pasta selecionada nao contem imagens .png")

print("")
print("Iniciando LBP")
for imagem in arquivosImagem:
	print("\tProcessando imagem " + imagem)
	try:
		A = color.rgb2gray(imread(os.path.join(caminhoEntrada, imagem)))
	except IOError as err:
		print("Erro na leitura da imagem ", imagem, ": ", err)
	
	# Filtros detectores de borda Roberts e Sobel:
	# (podem ser tirados, o HOG consegue funcionar sem eles)
	
	# a1 = roberts(A) # algoritmo de Roberts para deteccao de borda
	# a1 = sobel(A) # algoritmo de Sobel para deteccao de borda
	
	v = local_binary_pattern(A, 8, 64, method='default')

	# trocando a extensao .png por .txt:
	saida = imagem[:-3]
	saida = saida + "txt"

	try:
		caminhoSaida = os.path.join(caminhoEntrada,"LBP")
		if not os.path.exists(caminhoSaida):
			os.makedirs(caminhoSaida)
	except OSError as err:
		print("Erro de acesso a pasta de saida: ", err)

	try:
		f = open(os.path.join(caminhoSaida, saida), 'w')
		for x in np.nditer(v):
			f.write(str(x) + "\n")
	except IOError as err:
		print("Erro na escrita do arquivo ", saida, ": ", err)