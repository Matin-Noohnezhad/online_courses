#include <stm32f10x.h>
#include "delay.h"

void buzzer(){
	GPIOB->ODR|= (1<<10);
	delay_ms(200);		/* wait 200ms */
	GPIOB->ODR&= ~(1<<10);
}

void buzzOn(){
	GPIOB->ODR|= (1<<10);
}

void buzzOff(){
	GPIOB->ODR&= ~(1<<10);
}
