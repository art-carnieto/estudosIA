#include <stdio.h>
#include <malloc.h>
//#define nroNeuronios 2  //nroNeuronios + 1 (bias)
#define nroNeuronios 1  //nroNeuronios sem bias

//TODO: o que precisa arrumar neste codigo:
//1)Ate o primeiro resultado y_in esta correto
//Comeca a dar erro no calculo de erro e no
//alteracao (a t x), ai a segunda iteracao tambem
//sai errada
//
//2)Colocar as epocas (talvez um loop while ou for?)
//
//3)Fazer algum esquema de leitura de arquivos
//com o File e fscanf e escrita da resposta num .txt
//com o fprintf
//
//(opcional)4)Melhorar a leitura do codigo separando
//etapas do treinamento em funcoes diferentes

//Neste momento este codigo tenta simular o teste
//de mesa do Perceptron Simples, no slide 34 do
//arquivo RedesNeurais.pdf

void imprimeMatriz(int lin, int col, float m[lin][col]){
    int linha, coluna;
    for(linha=0; linha<lin; linha++){
        for(coluna=0; coluna<col; coluna++)
            printf("%f ", m[linha][coluna]);
        printf("\n");
    }
}

void multMatriz(int linA, int colA, int linB, int colB,
                float A[linA][colA], float B[linB][colB], float C[linA][colB]){
    if(colA != linB) {
        printf("DEU RUIM!\n");
        return;       //matriz nao pode multiplicar
    }

    int linC = linA;
    int colC = colB;

    int linha, coluna, i;
    float somaprod;

    for(linha=0; linha<linA; linha++){
        for(coluna=0; coluna<colB; coluna++){
            somaprod=0;
            for(i=0; i<colA; i++)
                somaprod += A[linha][i] * B[i][coluna];
            C[linha][coluna] = somaprod;
        }
    }
}

int main() {
    //os primeiro valor (posicao 0) eh do bias
    //float valoresX[][nroNeuronios] = {{0, 1}};

    //a primeira linha eh do bias
    //float pesosX[][nroNeuronios] = {{0, 0},
    //                                {0, 0.5}};

    ////////////////SEM BIAS//////////////////
    float valoresX[][nroNeuronios] = {{1}};
    float pesosX[][nroNeuronios] = {{-0.5}};

    float bias = 0;
    float pesosBias[][nroNeuronios] = {{0}};

    float saida[1][nroNeuronios];

//    printf("Bias + Neuronios:\n");
    printf("Neuronios:\n");
    imprimeMatriz(1, nroNeuronios, valoresX);
    printf("\n");

//    printf("Pesos (bias + neuronios):\n");
    printf("Pesos (somente neuronios):\n");
    imprimeMatriz(nroNeuronios, nroNeuronios, pesosX);
    printf("\n");

    printf("Bias:\n%f\n\n", bias);

    printf("Pesos dos bias:\n");
    imprimeMatriz(1, nroNeuronios, pesosBias);
    printf("\n");

    multMatriz(1, nroNeuronios, nroNeuronios, nroNeuronios, valoresX, pesosX, saida);

    int i, j;
    if(bias != 0){
        for(i=0; i<nroNeuronios; i++)
            saida[0][i] *= pesosBias[0][i];
    }

    printf("Saida (y_in):\n");
    imprimeMatriz(1, nroNeuronios, saida);
    printf("\n");

    ////////////////////////////////////////////////////////////////////////////////

//    float respEsperada[][nroNeuronios] = {{0, -1}};
    float respEsperada[][nroNeuronios] = {{-1}};
    float calcErro[1][nroNeuronios];

    for(i=0; i<nroNeuronios; i++)
        calcErro[0][i] = respEsperada[0][i] - saida[0][i]; //TODO rever aqui, nao tenho certeza

    printf("Matriz de calculo do erro:\n");
    imprimeMatriz(1, nroNeuronios, calcErro);
    printf("\n");

    ////////////////////////////////////////////////////////////////////////////////

    float alpha = 1.0;      //taxa de aprendizagem
    for(i=0; i<nroNeuronios; i++)
        calcErro[0][i] *= alpha;

    printf("alteracao (a t x):\n");
    imprimeMatriz(1, nroNeuronios, calcErro);
    printf("\n");

    for(i=0; i<nroNeuronios; i++)      //recalculo dos pesos
        for(j=0; j<nroNeuronios; j++)
            pesosX[i][j] += calcErro[0][i];

    printf("Pesos recalculados (dos neuronios):\n");
    imprimeMatriz(nroNeuronios, nroNeuronios, pesosX);
    printf("\n");

    ////////////////////////////////////////////////////////////////////////////////

    float saida2[1][nroNeuronios];
    multMatriz(1, nroNeuronios, nroNeuronios, nroNeuronios, saida, pesosX, saida2);

    if(bias != 0){
        for(i=0; i<nroNeuronios; i++)
            saida2[0][i] *= pesosBias[0][i];
    }

    printf("Saida2 (y_in):\n");
    imprimeMatriz(1, nroNeuronios, saida2);
    printf("\n");

    return 0;
}
