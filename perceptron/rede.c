#include "IATypes.h"
//#include "DriverIO.h"

typedef struct{

	Matriz* X;
	Matriz* v;

	
	Matriz* Z; //só usado quando
	valor bz; //bias do nivel Z
	Matriz* w; //ter 3 niveis

	Matriz* Y;
	valor by; //bias do nivel Y

}Rede;

// ------------------------------------------------------

/*
	definição de respostas (binaria 1,0 / bipolar 1,-1)
*/

valor valorAlto = 1.0;
valor valorBaixo = 0.0;

// ------------------------------------------------------

void setaFxDegrau(int x){
	// 0 = binaria
	// -1 = bipolar
	if( x == 0){
		valorAlto = 1;
		valorBaixo = 0;
	}else if(x == -1){
		valorAlto = 1;
		valorBaixo = -1;
	}
}

float fxAtivacaoDegrau(valor theta,valor Valor){
	if(Valor>theta) return valorAlto;
	else /*(Valor<theta)*/ return valorBaixo;
}

//pensar em mudar o preenchimento base dos pesos
float geraValorDefault(int linha,int coluna){
	return 0.95;
}

float geraValor(int linha,int coluna){
	return geraValorDefault(linha,coluna);
}

Rede* criarRede(int* niveis,int tam){
	Rede* r = (Rede*) malloc(sizeof(Rede));
	//modo de preenchimento dos pesos iniciais

	if(tam == 2){ //entrada, processamento / saida

		r->X = criaMatriz(niveis[0],1);
		// verificar orientação

		r->w = criaMatriz(niveis[0],niveis[1]);
		// verificar orientação

		r->Y = criaMatriz(niveis[1],1);
		// verificar orientação


		//preenchimento de w
		int i,j;
		for(i = 1;i<=r->w->linhas;i++){
			for(j = 1;j<=r->w->colunas;j++){
				setaValor(r->w,i,j,geraValor(i,j));
			}
		}

	}else if(tam == 3){  //nivel oculto

		r->X = criaMatriz(niveis[0],1);
		// verificar orientação

		r->v = criaMatriz(niveis[0],niveis[1]);
		// verificar orientação

		r->Z = criaMatriz(niveis[1],1);
		// verificar orientação

		r->w = criaMatriz(niveis[1],niveis[2]);
		// verificar orientação

		r->Y = criaMatriz(niveis[2],1);
		// verificar orientação


		//preenchimento de v
		int i,j;
		for(i = 1;i<=r->v->linhas;i++){
			for(j = 1;j<=r->v->colunas;j++){
				setaValor(r->v,i,j,geraValor(i,j));
			}
		}

		//preenchimento de w
		for(i = 1;i<=r->w->linhas;i++){
			for(j = 1;j<=r->w->colunas;j++){
				setaValor(r->w,i,j,geraValor(i,j));
			}
		}


	}
	return r;
}

// metodos de perceptron simples
void PS(Rede* r,int epocas, valor limiarErro, valor taxaDeAprendizado, Matriz* target){
	int epoca;
	for(epoca = 0; epoca < epocas ; epoca++){
		
		/**
			execução da rede
		*/
		
		r->Y = multMatriz(r->X, r->v);
		int i = 0;
		//pode dar erro
		//aplicar função de ativação
		for (i = 1; i < r->Y->colunas; i++) setaMatriz(r->Y,1,i,fxAtivacaoDegrau(1,retornaMatriz(r->Y,1,i,NULL) + r->by));

		/**
			calculo do erro
		*/

		


		/**
			back propagation
		*/

	}
}

//multilayer perceptron
void MLP(Rede* r,int epocas, valor limiarErro, valor taxaDeAprendizado, Matriz* target){


}




int main(){

	int redeDefault[] = {2,4};

	Rede* r = criarRede(redeDefault,2);

	//printf("l1: %i l2: %i l3: %i\n",r->X->linhas,r->Z->linhas,r->Y->linhas);

	Matriz* targetOR = criaMatriz(4,1);

	setaValor(targetOR,1,1,0);
	setaValor(targetOR,2,1,0);
	setaValor(targetOR,3,1,0);
	setaValor(targetOR,4,1,1);

	Matriz* targetAND = criaMatriz(4,1);

	setaValor(targetAND,1,1,0);
	setaValor(targetAND,2,1,1);
	setaValor(targetAND,3,1,1);
	setaValor(targetAND,4,1,1);

	return 0;
}
