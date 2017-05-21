#coding: utf-8
from skimage.io import imread
from skimage import feature, color
import numpy as np
import os
import sys
import pickle

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

def main(argv):

	if(len(argv) < 3):
		print("Numero errado de argumentos!")
		print("Usagem do lbp.py:")
		print("argumento-01: Inteiro que define o numero de pontos a ser considerado em cada raio de analise do LBP")
		print("argumento-02: Inteiro que define o raio de analise do LBP em pixels ")
		print("argumento-03: (opcional) '-verbose' (sem aspas): prints para debugging")
		return

	numPoints = int(argv[1]) # pixels por celula, definida pelo usuario
	radius = int(argv[2]) # orientations, definida pelo usuario

	verbose = False	# opcao de entrada do programa "-verbose" coloca prints para debugging
	if(len(argv) == 4):
		if(argv[3] == "-verbose"):
			verbose = True

	try:
		# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo lbp.py
		caminhoEntrada = "/home/arthur/SI/IA/EP/dataset1/treinamento" # pasta selecionada pelo usuario
		arquivosPasta = os.listdir(caminhoEntrada)
	except OSError as err:
		print("Erro no acesso a pasta com as imagens de entrada: ",err)
		return

	arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))

	if len(arquivosImagem) == 0:
		print("Pasta selecionada nao contem imagens .png")
		return

	i = 1
	existePasta = True
	try:
		while(existePasta == True):
			caminhoSaida = os.path.join(caminhoEntrada,"LBP" + str(i))
			if os.path.exists(caminhoSaida):
				i += 1
			else:
				existePasta = False
		os.makedirs(caminhoSaida)
	except OSError as err:
		print("Erro de acesso a pasta de saida: ", err)
		return

	for imagem in arquivosImagem:
		if (verbose == True):
			print("\tProcessando imagem " + imagem)

		try:
			A = color.rgb2gray(imread(os.path.join(caminhoEntrada, imagem)))
		except IOError as err:
			print("Erro na leitura da imagem ", imagem, ": ", err)
			continue

		v = describe(A, numPoints, radius)

		# trocando a extensao .png por .txt:
		saida = imagem[:-3]
		saida = saida + "txt"

		try:
			f = open(os.path.join(caminhoSaida, saida), 'w')
			for x in np.nditer(v):
				f.write(str(x) + "\n")
		except IOError as err:
			print("Erro na escrita do arquivo ", saida, ": ", err)

	try:
		arqConfig = open(os.path.join(caminhoSaida,"configExtrator.dat"), "wb")
	except IOError as err:
		print("Erro na escrita do arquivo configExtrator.dat", err)
		return

	data = (numPoints, radius)
	pickle.dump(data, arqConfig)
	arqConfig.close()

	try:
		arqLog = open(os.path.join(os.getcwd(),"logExtratores.txt"), "a")
	except IOError as err:
		print("Erro na escrita do arquivo logExtratores.txt", err)
		return

	texto = ["LBP" + str(i) + ": " + caminhoSaida + "\n",
			 "numPoints = " + str(numPoints) + "\n",
			 "radius = " + str(radius) + "\n\n"
	]
	for linha in texto:
		arqLog.write(linha)
	arqLog.close()

if __name__ == "__main__":
    main(sys.argv)