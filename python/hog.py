#coding: utf-8
# script para passar todas as imagens no filtro
# HOG e escreve-las em uma pasta na saida

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import color
import os

np.set_printoptions(threshold=np.nan) # corrige o tamanho maximo do print na saida padrao de um ndarray

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo hog.py
	caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento" # pasta selecionada pelo usuario
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))

# ppc = [128, 64, 32, 16, 8, 4, 2] # ppc = pixels por celula, para varios casos
ppc = [32] # ppc = pixels por celula, somente para 32x32

if len(arquivosImagem) == 0:
	print("Pasta selecionada nao contem imagens .png")

for i in ppc:
	print("")
	print("Iniciando HOG com pixels_per_cell = " + str(i) + "x" + str(i))
	for imagem in arquivosImagem:
		print("\tProcessando imagem " + imagem)
		try:
			A = color.rgb2gray(imread(os.path.join(caminhoEntrada, imagem)))
		except IOError as err:
			print("Erro na leitura da imagem ", imagem, ": ", err)

		v, B = hog(A,orientations=10, pixels_per_cell=(i, i),
			cells_per_block=(1, 1), visualise=True)

		'''
		# plots para mostrar as imagens geradas:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

		ax1.axis('off')
		ax1.imshow(A, cmap=plt.cm.gray)
		ax1.set_title('Imagem de Entrada')
		ax1.set_adjustable('box-forced')

		ax2.axis('off')
		ax2.imshow(B)
		ax2.set_title('HOG')
		ax2.set_adjustable('box-forced')

		plt.show()
		'''
		
		# trocando a extensao .png por .txt:
		saida = imagem[:-3]
		saida = saida + "txt"

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