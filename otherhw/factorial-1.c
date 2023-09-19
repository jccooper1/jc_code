/*
 *	
 *  Factorial!: Computes the factorial of a positive integer
 *
 */

#include <stdio.h>

int main()
{
    /* Initialization */
    int factorial;   /* input to be entered by the user */
    int result;      /* result,  factorial! */
    while(1)
    {
        printf("Please enter a number: ");
        scanf("%d", &factorial);
        if(factorial<0 || factorial>12){
            printf("The input is not acceptable, try again.");
        }
        else
        {
            int i;
            /* Compute factorial */
            result = 1;
            for (i = factorial; i > 0; i = i-1) {
                result *= i;
            }
            /* Print the answer */
            printf("%d\n", result);
            break;
            }
        }
        return 0;
}