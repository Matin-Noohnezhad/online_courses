#include <stdio.h>
#include <stdbool.h>


extern bool  is_substring(char*, char*);

int main() {

	char str1[50];
	char str2[50];


	printf("please input string1: ");
	fgets(str1, 50, stdin);
	printf("please input string2: ");
	fgets(str2, 50, stdin);
	printf("the legnth of str1 is %d\n", strlen(str1));
	printf("the legnth of str2 is %d\n", strlen(str2));
	bool a = is_substring(&str1, &str2);
	if (a == true) {
		printf("string2 is a substring of string1");
	}
	else {
		printf("string2 is not a substring of string1");
	}
	return 0;
}


