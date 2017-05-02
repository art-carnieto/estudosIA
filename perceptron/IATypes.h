#include <stdio.h>
#include <stdbool.h>
#include <malloc.h>

typedef float valor;

typedef struct{
	int linhas;
	int colunas;
	valor** valores;

}Matriz;

Matriz* criaMatriz(int linhas, int colunas){
	Matriz* m = (Matriz*) malloc(sizeof(Matriz));
	m->linhas = linhas;
	m->colunas = colunas;

	m->valores = (valor**) malloc(sizeof(valor*) * linhas);

	int i;
	for(i = 0;i<linhas;i++){
		m->valores[i] = (valor*) malloc(sizeof(valor) * colunas);
	}

	return m;
}

bool setaValor(Matriz* m,int linha, int coluna, valor novoValor){

	if(m->linhas < linha || linha < 1) return false;
	if(m->colunas < coluna || coluna < 1) return false;
	/** convenção da matriz */

	m->valores[linha-1][coluna-1] = novoValor;

	return true;
}

valor retornaValor(Matriz* m,int linha, int coluna, bool* ok){
	//lembrar que *ok não pode ser null, pensar em outra solução
	if(m->linhas < linha || linha < 1){	*ok = false; return 0.;	}
	if(m->colunas < coluna || coluna < 1){ *ok = false;	return 0.; }
	/** convenção da matriz */

	*ok = true;
	return m->valores[linha-1][coluna-1];
	/** convenção da matriz */

	/*
		bool ok;
		valor resp = retornaValor(m,1,-1,&ok);
		printf("%s , %f \n", ok ? "true" : "false",resp);
	*/

}

void mostraMatriz(Matriz* m){
	int i;
	for(i = 0;i<m->linhas;i++){
		printf("{");
		int j;
		for(j = 0;j<m->colunas;j++){
			printf("%f",m->valores[i][j]);
			if(j != m->colunas-1) printf(", ");
		}
		printf("}\n");
	}

}

Matriz* multMatriz(Matriz* m1, Matriz* m2){
    if(m1->colunas != m2->linhas) {
        printf("DEU RUIM!\n");
        return NULL;       //matriz nao pode multiplicar
    }

    Matriz* resp = criaMatriz(m1->linhas,m2->colunas);

    int linha, coluna, i;
    float somaprod;

    for(linha=0; linha< m1->linhas ; linha++){
        for(coluna=0; coluna<m2->colunas ; coluna++){
            somaprod=0;
            for(i=0; i<m1->colunas ; i++)
                somaprod += m1->valores[linha][i] * m2->valores[i][coluna];
            setaValor(resp,linha+1,coluna+1,somaprod);
        }
    }

    return resp;
}

Matriz* reversa(Matriz* m){
	Matriz* resp = criaMatriz(m->colunas,m->linhas);
	float aux;
	int i,j;
	for(i = 0; i < m->linhas; i++)
	    for(j = 0; j < m->colunas; j++){
	    	resp->valores[j][i] = m->valores[i][j];
	 	}
	return resp;
}


/**
	testes

	Matriz* m = criaMatriz(3,5);
	printf("seta: %s \n",setaValor(m,2,1,5.3) ? "true" : "false");
	bool resp = false;
	printf("retorna: %f \n",retornaValor(m,2,1,&resp));
	printf("-------------\n");
	mostraMatriz(m);

	-------------------------------------------------------------------------

	Matriz* a = criaMatriz(3,1);

    mostraMatriz(a);
    printf("--------------------\n");

	printf("seta: %s \n",setaValor(a,1,1,1.) ? "true" : "false");
	printf("seta: %s \n",setaValor(a,2,1,2.) ? "true" : "false");
	printf("seta: %s \n",setaValor(a,3,1,3.) ? "true" : "false");

	Matriz* b = criaMatriz(1,3);
    mostraMatriz(b);
    printf("--------------------\n");
	printf("seta: %s \n",setaValor(b,1,1,1.) ? "true" : "false");
	printf("seta: %s \n",setaValor(b,1,2,2.) ? "true" : "false");
	printf("seta: %s \n",setaValor(b,1,3,3.) ? "true" : "false");

	mostraMatriz(a);
	printf("----------------\n");
	mostraMatriz(b);

*/
