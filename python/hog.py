#coding: utf-8
# script para passar todas as imagens no filtro
# HOG e escreve-las em uma pasta na saida

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import color
import os
import pickle
import sys

np.set_printoptions(threshold=np.nan) # corrige o tamanho maximo do print na saida padrao de um ndarray

def main(argv):

	if(len(argv) < 4):
		print("Numero errado de argumentos!")
		print("Usagem do hog.py:")
		print("argumento-01: Inteiro que define o numero de pixels por celula")
		print("argumento-02: Inteiro que define o numero de orientacoes")
		print("argumento-03: Inteiro que define o numero de celulas por bloco")
		print("argumento-04: (opcional) '-verbose' (sem aspas): prints para debugging")
		return

	ppc = int(argv[1]) # pixels por celula, definida pelo usuario
	ori = int(argv[2]) # orientations, definida pelo usuario
	cpb = int(argv[3]) # cells_per_block, definida pelo usuario

	verbose = False	# opcao de entrada do programa "-verbose" coloca prints para debugging
	if(len(argv) == 5):
		if(argv[4] == "-verbose"):
			verbose = True

	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo hog.py
	pastaBase = "/home/arthur/SI/IA/EP" # pasta selecionada pelo usuario

	caminhos = []
	#parte 1
	#caminhos.append(os.path.join(pastaBase, "dataset1", "testes"))
	#caminhos.append(os.path.join(pastaBase, "dataset1", "treinamento"))
	
	#parte 2
	caminhos.append(os.path.join(pastaBase, "dataset2", "testes"))
	caminhos.append(os.path.join(pastaBase, "dataset2", "treinamento"))

	for caminhoEntrada in caminhos:
		try:
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
				caminhoSaida = os.path.join(caminhoEntrada,"HOG" + str(i))
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

			v, B = hog(A,orientations=ori, pixels_per_cell=(ppc, ppc),
				cells_per_block=(cpb, cpb), visualise=True)

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
				f = open(os.path.join(caminhoSaida, saida), 'w')
				# f.write(str(v)[2:-1])	# tem esse corte [2:-1] para tirar '[' e ']' da saida
				for x in np.nditer(v):
					f.write(str(x) + "\n")
			except IOError as err:
				print("Erro na escrita do arquivo ", saida, ": ", err)

		try:
			arqConfig = open(os.path.join(caminhoSaida,"configExtrator.dat"), "wb")
		except IOError as err:
			print("Erro na escrita do arquivo configExtrator.dat", err)
			return

		data = (ppc, ori, cpb)
		pickle.dump(data, arqConfig, protocol=2)
		arqConfig.close()

		try:
			arqLog = open(os.path.join(os.getcwd(),"logExtratores.txt"), "a")
		except IOError as err:
			print("Erro na escrita do arquivo logExtratores.txt", err)
			return

		texto = ["HOG" + str(i) + ": " + caminhoSaida + "\n",
				 "ppc = " + str(ppc) + "\n",
				 "ori = " + str(ori) + "\n",
				 "cpb = " + str(cpb) + "\n\n"
		]
		for linha in texto:
			arqLog.write(linha)
		arqLog.close()

if __name__ == "__main__":
    main(sys.argv)