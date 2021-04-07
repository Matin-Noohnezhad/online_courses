
#include  <stdio.h>

extern int func(int, int);

int main() {
	int a = 3;
	int b = 2;
	int c = func(a, b);
	printf("%d", c);
	return 0;
}
