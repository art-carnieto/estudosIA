#include <stdio.h>
#include <malloc.h>
//#include <openacc.h>
#include <stdbool.h>

typedef float valor;

typedef struct s{
	valor w;
	struct s* prox;
}Axonio;

typedef struct{

	valor y;
	Axonio* ligacoes;

}Neuronio;

typedef struct {
	Neuronio* neuronios;
	valor bias;
} Nivel;

typedef struct {
	Nivel* niveis;
}Rede;

// ---------------------------------

// ---------------------------------

Rede* criarRede(int niveis){
	Rede* r = (Rede*) malloc(sizeof(Rede));
	r->niveis = NULL;
	r->niveis = (Nivel*) malloc(sizeof(Neuronio)*niveis);

	for(int i = 0;i<niveis;i++){
		r->niveis[i] = NULL;
	}

	return r;

}

Nivel* criarNivel(int qtde){
	Nivel* n = (Nivel*) malloc(sizeof(Nivel));
	n->bias = 0;
	n->neuronios = (Neuronio*) malloc(sizeof(Neuronio)*qtde);

	for(int i = 0;i<qtde;i++){
		r->neuronios[i] = NULL;
	}

	return n;
}

Neuronio* criarNeuronio(int x){
	Neuronio* n = (Neuronio*) malloc(sizeof(Neuronio));
	n->valor = x;
	n->ligacoes = (Axonio*) malloc(sizeof(Axonio))
}



int main(){

	Rede* r = criarRede(17);
	


	return 0;
}