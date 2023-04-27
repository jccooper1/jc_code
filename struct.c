
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
 
int main()
{
	char str1[10] = "hello world";
	char str2[10] = "*********";
	printf("%s\n", strcpy(str1,str2));
 
	return 0;
}
