#coding: utf-8
#script para mudar a resolucao das imagens
from PIL import Image
import os

try:
	# caminhoEntrada = os.getcwd() # os.getcwd ==> pasta atual do arquivo resolucao.py
	caminhoEntrada = "/home/juuullyanne/Área de Trabalho/IA/dataset1/treinamento" # pasta selecionada pelo usuario
	arquivosPasta = os.listdir(caminhoEntrada)
except OSError as err:
	print("Erro no acesso a pasta com as imagens de entrada: ",err)

arquivosImagem = list(filter(lambda k: '.png' in k, arquivosPasta))

if len(arquivosImagem) == 0:
	print("Pasta selecionada nao contem imagens .png")

res = [8, 16, 32, 64]

print("")
print("Iniciando mudancas de resolucoes")
for imagem in arquivosImagem:
	print("\tProcessando imagem " + imagem)
	try:
		im = Image.open(os.path.join(caminhoEntrada, imagem))
	except IOError as err:
		print("Erro na leitura da imagem ", imagem, ": ", err)
	
	for i in res:
		im_resized = im.resize((i ,i), Image.ANTIALIAS)
		caminhoSaida = ""
		try:
			caminhoSaida = os.path.join(caminhoEntrada,"resolucoes")
			if not os.path.exists(caminhoSaida):
				os.makedirs(caminhoSaida)
		except OSError as err:
			print("Erro de acesso a pasta de saida: ", err)

		try:
			caminhoSaida = os.path.join(caminhoSaida,str(str(i)+"x"+str(i)))
			if not os.path.exists(caminhoSaida):
				os.makedirs(caminhoSaida)
		except OSError as err:
			print("Erro de acesso a pasta de saida: ", err)

		try:
			im_resized.save(os.path.join(caminhoSaida, imagem), dpi=(i,i))
		except IOError as err:
			print("Erro na escrita do arquivo ", imagem, ": ", err)