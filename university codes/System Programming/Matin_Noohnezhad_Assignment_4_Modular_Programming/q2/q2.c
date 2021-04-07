#include <stdio.h>
#include "q2.h"

//extern int myadd(int, int);
//extern void myswap(int, int);

int main() {
	int i = 3;
	int j = 4;
	int result = myadd(i , j);
	printf("the result of adding two number is %d\n", result);
	myswap(&i , &j);
	printf("the value of i after swap is %d\n", i);
	printf("the value of j after swap is %d\n", j);
	return 0;
}