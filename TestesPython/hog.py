# script para passar todas as imagens no filtro
# HOG e escreve-las em uma pasta na saida

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import color
from skimage.filters import roberts
import os

np.set_printoptions(threshold=np.nan) # corrige o tamanho maximo do print na saida do ndarray

# aonde a pasta com as imagens esta
# os.getcwd ==> pasta atual do arquivo hog.py
# pode mudar para outra string aonde estejam as imagens
try:
	caminhoEntrada = os.getcwd()
	#caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento"
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)
# print("arquivos:")
# print(arquivosPasta)
# print("")
arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))
# print("somente imagens:")
# print(arquivosImagem)

# neste caso utiliza-se o detector de bordas Robert
# neste site possuem outros algoritmos de deteccao de borda:
# http://scikit-image.org/docs/dev/auto_examples/edges/plot_edge_filter.html

ppc = [128, 64, 32, 16, 8, 4, 2] # ppc = pixels por celula

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
		a1 = roberts(A)
		v, B = hog(a1,orientations=8, pixels_per_cell=(i, i),
			cells_per_block=(1, 1), visualise=True)

		# testes para mostrar as imagens geradas:
		# imshow(A)
		# plt.show()
		# imshow(a1)
		# plt.show()
		# imshow(B)
		# plt.show()

		# saida = string imagem menos os 3 ultimos caracteres
		saida = imagem[:-3]
		saida = saida + "txt"
		# print("saida = ",saida)

		try:
			caminhoSaida = os.path.join(caminhoEntrada,"HOG " + str(i))
			if not os.path.exists(caminhoSaida):
				os.makedirs(caminhoSaida)
		except OSError as err:
			print("Erro de acesso a pasta de saida: ", err)

		try:
			f = open(os.path.join(caminhoSaida, saida), 'w')
			f.write(str(v)[2:-1])
		except IOError as err:
			print("Erro na escrita do arquivo ", saida, ": ", err)