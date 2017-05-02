#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> //para leitura de pasta e arquivos

int main()
{

//  listar todos os arquivos de uma pasta ==> http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
//  armazenar os nomes dos arquivos ==> http://stackoverflow.com/questions/1088622/how-do-i-create-an-array-of-strings-in-c
//  append strings em C ==> http://stackoverflow.com/questions/308695/how-to-concatenate-const-literal-strings-in-c
//  tratamento de erros de IO ==> https://www.tutorialspoint.com/cprogramming/c_error_handling.htm
//  leitura/escrita de doubles em C ==>http://stackoverflow.com/questions/4264127/correct-format-specifier-for-double-in-printf

    DIR *dir;
    struct dirent *ent;
    char dirEntrada[] = "/home/arthur/SI/IA/EP/dataset1/treinamento/HOG 16/";
    FILE* f = fopen("teste.txt", "w");

    if(!f){
        perror ("Erro de escrita do arquivo teste.txt");
        exit(EXIT_FAILURE);
    }

    char arquivos[3000][50];    //3000 arquivos com 50 caracteres cada nome
    int i = 0;

    if ((dir = opendir (dirEntrada)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL && i < 3000) {
            if (ent->d_type == DT_REG){
                strcpy(arquivos[i], ent->d_name);
                fprintf(f, "arquivos[%d] = %s\n", i, arquivos[i]);
                if(i >= 0 && i < 4)printf("arquivos[%d] = %s\n", i, arquivos[i]);
                i++;
            }
        }
        closedir (dir);
    }

    else {
        /* could not open directory */
        perror ("Erro de acesso a pasta com arquivos de entrada da rede");
        exit(EXIT_FAILURE);
    }

//    teste maroto pq tava dando tudo errado antes:
    FILE* testeArq = fopen("testeIntegridade.txt", "w");

    if(!testeArq){
        perror ("Erro de escrita do arquivo testeIntegridade.txt");
        exit(EXIT_FAILURE);
    }

    for(i=0; i<3000; i++){
        fprintf(testeArq, "arquivos[%d] = %s\n", i, arquivos[i]);
    }
    fclose(testeArq);

//    teste de leitura para o primeiro arquivo (arquivos[0])
    char caminhoCompleto[500];
    snprintf(caminhoCompleto, sizeof(caminhoCompleto), "%s%s", dirEntrada, arquivos[0]);
    printf("caminhoCompleto = %s", caminhoCompleto);
    printf("\n\n");
    FILE* arq = fopen(caminhoCompleto, "r");
    FILE* arq2 = fopen("testeConteudo.txt", "w");

    if(!arq){
        perror("Erro de leitura do .txt contendo a imagem");
        exit(EXIT_FAILURE);
    }

    if(!arq2){
        perror("Erro de escrita do testeConteudo.txt");
        exit(EXIT_FAILURE);
    }

    i=0;
    double teste;
    while(!feof(arq)){
        fscanf(arq, "%lf ", &teste);
        fprintf(arq2, "%d = %.8f\n", i, teste);    //8 casas decimais sao o suficiente? TESTAR!
        i++;
    }
    fclose(arq);
    fclose(arq2);

    return 0;
}
