#include <stm32f10x.h>
#include "delay.h"

/* The function checks if a key is pressed or not */
/* Returns: 0: Not pressed, 1: Pressed */
uint8_t keypad_kbhit()
{
	GPIOB->ODR &= ~(0x0F<<12); /* make all rows ones */
	delay_us(1);
	if(((GPIOB->IDR>>6)&0xF) == 0xF)
		return 0;
	else
		return 1;
}

/* The function returns the pressed key. */
/* It returns 0 if no key is pressed. */

uint8_t keypad_getkey()
{
	uint8_t keypadLookup[16]={'1','2','3','A','4','5','6','B','7','8','9','C','*','0','#','D'};
	const uint32_t rowSelect[4]={
		0x1000E000, /* Row3-Row0 = 1110 */
		0x2000D000, /* Row3-Row0 = 1101 */
		0x4000B000, /* Row3-Row0 = 1011 */
		0x80007000};/* Row3-Row0 = 1011 */	
				
	if(keypad_kbhit() == 0)	/* if no key is pressed */
		return 0;
	
	for(int r = 0; r <= 3; r++) /* rows 0 to 3 */
	{
		GPIOB->BSRR = rowSelect[r]; /* ground row r and make the others high */
		delay_us(1);	/* wait for the columns to beupdated */
		uint8_t cols = (GPIOB->IDR>>6)&0xF;
		
		switch(cols) 
		{
			case 0x0E: return keypadLookup[r*4+0]; /* col0 is low */
			case 0x0D: return keypadLookup[r*4+1]; /* col1 is low */
			case 0x0B: return keypadLookup[r*4+2]; /* col2 is low */
			case 0x07: return keypadLookup[r*4+3]; /* col3 is low */
		}
	}
	
	return 0;
}
