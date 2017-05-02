#include "IATypes.h"
#include <string.h>

typedef struct s{
	valor val;
	struct s *prox;
}NO;

typedef struct r{
	NO* linha;
	struct r *prox;
}NOm;

Matriz* leMatriz(char* end){
	
	FILE *arq;
	
	arq = fopen(end, "r");

	if(arq == NULL) printf("Erro, nao foi possivel abrir o arquivo\n");
	

	int linhas = 0;
	int colunas = 0;

	NOm* mat = NULL;
	NOm* linha = NULL;
	NO* naLinha = NULL;

	int ch;
	/*
	while(!feof(arq)){
		while(fscanf(arq,"%i[^\n]",ch)){
			printf("%i ", ch);
		}
		
		//if(ch != "\n") printf("%c ", *ch);

	}
	
	*/

	/**
	    while( (ch=fgetc(arq))!= EOF ){


			putchar(ch);
			

		}
	*/
			
	fclose(arq);
	
	return 0;
}

