#include <stdio.h>
#include <stdlib.h>

#define MAX_CHARS 200

int main(void){

  FILE *fin=NULL, *fout=NULL;
  char str[MAX_CHARS], buff[MAX_CHARS];
  int i=0, num=0;
  
  
  /* opening file for reading */
  fin = fopen("train_preprocessed.csv" , "r");
  //fout = fopen("train_for_lr.csv", "w");
  if(fin == NULL) {
    perror("Error opening file");
    return(-1);
  }

  while( fgets (str, MAX_CHARS, fin)!=NULL ) {
    /* writing content to stdout */
    for (i=0; i < MAX_CHARS; i++){
      if(str[i] == '\t'){
	buff[i] = '\t';
	break;
      }
      buff[i] = str[i];
    }
    buff[i+1] = '\0';
    puts(buff);
    num = atoi(&str[i+1]);
    printf("%d\n",num);
  }
  fclose(fin);
  //fclose(fout);
  return 0;
}
