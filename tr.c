#include<stdio.h>
int main() {
    int a = 1;
    char b = 'B';
    float c = 3.3;

    void * ptr = &a;
    int *a2 = (int*) ptr;
    printf("The number is: %d \n", *a2);
    
    ptr = &b;
    char *b2 = (char*) ptr;
    printf("The character is: %c \n", *b2);

    ptr = &c;
    float *c2 = (float*) ptr;
    printf("The float is: %f \n", *c2);
    return 0;
}