#coding: utf-8
from skimage.io import imread
from skimage import feature, color
import numpy as np
import os

def describe(image, numPoints, radius, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,
			radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, numPoints + 3),
			range=(0, numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo lbp.py
	caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento" # pasta selecionada pelo usuario
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))

if len(arquivosImagem) == 0:
	print("Pasta selecionada nao contem imagens .png")

numPoints = 24
radius = 8

print("")
print("Iniciando LBP")
for imagem in arquivosImagem:
	print("\tProcessando imagem " + imagem)

	try:
		A = color.rgb2gray(imread(os.path.join(caminhoEntrada, imagem)))
	except IOError as err:
		print("Erro na leitura da imagem ", imagem, ": ", err)

	v = describe(A, numPoints, radius)

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