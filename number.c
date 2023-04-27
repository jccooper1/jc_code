#include<stdio.h>
int main()
{
   int a,b,z;
   char operator='&';
   printf("Enter '&' or '|' :");
   
   printf("A B Z\n");

   for(a=0; a<=1; a++){
    for(b=0;b<=1;b++){
        if(operator=="&"){
            z=a&b;
        }else{
            z=a|b;
        }
        printf("%d %d %d\n",a,b,z);
    }
   }
}