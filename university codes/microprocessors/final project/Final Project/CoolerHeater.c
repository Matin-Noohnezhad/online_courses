#include <stm32f10x.h>

void first_cooler(uint16_t u){
	if(u==0){
		//off
		GPIOA->ODR&= ~(1<<11);
	}else{
		//on
		GPIOA->ODR|= (1<<11);
	}
}
void second_cooler(uint16_t u){
	if(u==0){
		//off
		GPIOA->ODR&= ~(1<<12);
	}else{
		//on
		GPIOA->ODR|= (1<<12);
	}
}
void first_heater(uint16_t u){
	if(u==0){
		//off
		GPIOB->ODR&= ~(1<<0);
	}else{
		//on
		GPIOB->ODR|= (1<<0);
	}
}
void second_heater(uint16_t u){
		if(u==0){
		//off
		GPIOB->ODR&= ~(1<<1);
	}else{
		//on
		GPIOB->ODR|= (1<<1);
	}
}
