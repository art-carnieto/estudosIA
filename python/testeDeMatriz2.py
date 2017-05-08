import numpy as np
from random import randint

def combinacaoDeEmbralhamento(qtdeLinhas):
	while True:
		a = randint(0,qtdeLinhas)
		b = randint(0,qtdeLinhas)
		if a == b :
			continue
		else:
			break
	resp = [[a,b]]
	return resp

	

def embaralhaMatriz(x):
	for i in range(qtdeLinhas):
		print(i)
	combinacaoDeEmbralhamento(x.shape[0])
	x[[1,0]] = x[[0,1]]


# MAIN

A = np.zeros(shape=(5,2))

print(A)
print("-------------")
print(A.shape)