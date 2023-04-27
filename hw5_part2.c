#include <stdio.h>
int main()
{
    unsigned int x,y,z,w,b,d,a,c;
    unsigned int g;
     
    /* Print header for K-map. */
    printf("         xy      \n");
    printf("     00 01 11 10 \n");
    printf("   ______________\n");
     
    /* row-printing loop */
    for (z = 0; 2 > z; z = z + 1) {
    
        for(a=0;2>a;a=a+1){
            w=a^z;
        printf("zw=%u%u | ", z,w);
        /* Loop over input variable b in binary order. */
            for (x = 0; 2 > x; x = x + 1) {
                
            /* Loop over d in binary order.*/
                for (d = 0; 2 > d; d = d + 1) {
                    y=x^d;
                    g=(~x&1)|(w&x&(~y)&z&1)|(y&(~z)&1);
                
                    printf("%u  ",g);
               
            }
        }
        printf("\n");
        }
            
    }
     
    return 0;
}
