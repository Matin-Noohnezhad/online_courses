#include <stm32f10x.h>
#include <stdio.h>

void usart1_init()
{
	USART1->CR1 = 0x200C;
	USART1->BRR = 7500;	 // 72MHz/9600bps = 7500	
}

void usart1_sendByte(unsigned char c)
{	
	while((USART1->SR&(1<<6)) == 0);  //wait until the TC flag is set 
	USART1->DR = c;
}

/* the function sends a zero ending string through USART1 */
void usart1_sendStr(char *str)
{
	while(*str != 0)
	{
		usart1_sendByte(*str);
		str++;
	}
}

/* The function sends a number through USART1 */
void usart1_sendInt(unsigned int i)
{
	char str[10];
	sprintf(str,"%d",i);
	
	usart1_sendStr(str);	
}

uint8_t usart2_recByte()
{
	while((USART2->SR&(1<<5)) == 0);	//wait until the RXNE flag is set
	
	return USART2->DR;	
}
