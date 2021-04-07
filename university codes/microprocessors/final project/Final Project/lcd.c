#include <stm32f10x.h>
#include <stdio.h>
#include "delay.h"

#define LCD_RS 0
#define LCD_EN 1

void lcd_putValue(unsigned char value)
{
	GPIOA->BRR = 0xF0; /* clear PA0-PA3 */
	GPIOA->BSRR = value&0xF0; /* put high nibble on PA0-PA3 */	
	
	GPIOA->ODR |= (1<<LCD_EN); /* EN = 1 for H-to-L pulse */
	delay_us(1);			/* make EN pulse wider */
	GPIOA->ODR &= ~ (1<<LCD_EN);	/* EN = 0 for H-to-L pulse */
	delay_us(100);			/* wait	*/

	GPIOA->BRR = 0xF0; /* clear PA0-PA3 */
	GPIOA->BSRR = (value<<4)&0xF0; /* put low nibble on PA0-PA3 */	
	
	GPIOA->ODR |= (1<<LCD_EN); /* EN = 1 for H-to-L pulse */
	delay_us(1);			/* make EN pulse wider */
  GPIOA->ODR &= ~(1<<LCD_EN);	/* EN = 0 for H-to-L pulse */
  delay_us(100);			/* wait	*/
}

void lcd_command(unsigned char cmd)
{
	GPIOA->ODR &= ~ (1<<LCD_RS);	/* RS = 0 for command */
	lcd_putValue(cmd);
}

void lcd_data(unsigned char data)
{
	GPIOA->ODR |= (1<<LCD_RS);	/* RS = 1 for data */
	lcd_putValue(data); 
}

void lcd_print(char * str)
{
  unsigned char i = 0;

	while(str[i] != 0) /* while it is not end of string */
	{
		lcd_data(str[i]); /* show str[i] on the LCD */
		i++;
	}
}

void lcd_gotoxy(unsigned char x, unsigned char y)
{
	const unsigned char firstCharAdr[]={0x80,0xC0,0x94,0xD4};
	lcd_command(firstCharAdr[y-1] + x - 1);
	delay_us(100);	
}

void lcd_init()
{
	GPIOA->ODR &= ~(1<<LCD_EN);	/* LCD_EN = 0 */
	delay_us(3000); 			/* wait 3ms */
	lcd_command(0x33);		/* send $33 for init. */
	lcd_command(0x32);		/* send $32 for init. */
	lcd_command(0x28);		/* init. LCD 2 line,5´7 matrix */
	lcd_command(0x0e);		/* display on, cursor on */
	lcd_command(0x01);		/* clear LCD */
	delay_us(2000);				/* wait 2ms */
	lcd_command(0x06);		/* shift cursor right */
}
