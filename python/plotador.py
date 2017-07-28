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

arquivoErros = open(os.path.join(os.getcwd(),'erro.txt'),'r')
caminhoSaida = os.getcwd()

arquivoErros.readline()

for i in range(5):
	errosTreino = []
	errosValidacao = []

	for line in arquivoErros:
		#print("linha = " + str(line))
		if line == "\n":
			break
		stringCortada = line.split(';')
		errosTreino.append(float(stringCortada[1]))
		errosValidacao.append(float(stringCortada[2]))

	plt.plot(errosTreino, linewidth=2, label='Erro de treinamento')
	plt.plot(errosValidacao, dashes=[2, 1], label='Erro de validacao')
	plt.title('Rodada ' + str(i))
	plt.ylabel('Erro')
	plt.xlabel('Epoca')
	nomePlot = 'erros_rodada' + str(i) + '.png'
	plt.legend()
	plt.savefig(os.path.join(caminhoSaida, nomePlot))
	plt.close()