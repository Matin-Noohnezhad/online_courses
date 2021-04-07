/* This program configures PA2(ADC_IN2) as the ADC input and sends
the result through usart1 */

#include <stm32f10x.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "keyboard.h"
#include "delay.h"
#include "uart.h"
#include "lcd.h"
#include "CoolerHeater.h"
#include "Buzzer.h"

volatile static unsigned int CURRENT_TEMP = 34;
volatile static unsigned int MAX_TEMP = 37;
volatile static unsigned int MIN_TEMP = 25;
volatile static unsigned int TARGET_TEMP = 32;

volatile static int COUNTER = 0;

volatile static unsigned int SEC = 0;
volatile static unsigned int MIN = 0;
volatile static unsigned int HOUR = 0;

//layer=1 (menu) layer=2 (inside menu)
volatile static unsigned int LAYER = 1;


static char* complete_hour ;
static char* menu_item;

#define ARR_SIZE  5

static char MENU[ARR_SIZE][14] = {"Set Temp", "set max temp","set min temp","Show Logs", "Delete Logs"};

void set_current_temp(uint16_t u );

void increase_sec(void);
void increase_min(void);
void increase_hour(void);

void show_time(void);

void show_menu(void);
void show_temp_menu(uint16_t u);

void ok_pressed(void);
void up_pressed(void);
void down_pressed(void);

void keyboard_input(void);

//interrupts
void SysTick_Handler(){
		usart1_sendStr("systick\n\r");	/* send temperature through serial */
	increase_sec();
	show_time();
}

int main()
{
	complete_hour = malloc(8); 
	menu_item = malloc(16);
	//
	RCC->APB2ENR |= 0xFC|(1<<9)|(1<<14); 	/* enable clocks for GPIO, ADC1 clock, and usart1 */

	//SysTick for increasing time
	SysTick->LOAD = 9000000-1;
	SysTick->CTRL = 0x03;
	/* init keypad pins */
	GPIOB->CRL = 0x88443433; /* PB6 and PB7 as inputs <<PB0, PB1 as output for heaters>> */
	GPIOB->CRH = 0x33334388; /* PB15-PB12 and PB10(buzzer) as outputs, PB8 and PB9 as inputs */
	GPIOB->ODR |= (0xF<<6); /* pull-up PB9-PB6 */
		//USART1 init.
	GPIOA->ODR |= (1<<10);  //pull-up PA10
	GPIOA->CRH = 0x444338B4; // RX1=input with pull-up, TX1=alt. func output <<PA11,PA12 output for coolers>>
	
	
	GPIOA->CRL = 0x33334033; 	/* PA2(ADC_IN2) as analog input -- <<PA0(RS), PA1(enable),PA4-7(data)  (LCD)(outputs)>>*/
	ADC1->CR2 = 1;	/* ADON = 1 (power-up) */
	
	ADC1->SMPR2 = 1<<6; /* SMPR2.SMP2 = 001 */
		
	delay_us(1);	/* wait 1us to make sure the adc module is stable */
	
	usart1_init(); /* initialize the usart1 */
	
	lcd_init();
	
	show_menu();
	//lcd_print("Hello World!"); /* show "Hello World!" */

while(1)
	{		
		keyboard_input();
		//
		ADC1->SQR3 = 2;	/* choose channel 2 as the input */
		ADC1->CR2 = 1;	/* ADON = 1 (start conversion) */
		while((ADC1->SR&(1<<1)) == 0); /* wait until the EOC flag is set */
		
		unsigned int temp = ADC1->DR*330/4096; /* read ADC1_DR and calculate temperature */
		set_current_temp(temp);
		//usart1_sendInt(temp);	/* send temperature through serial */
		//usart1_sendStr("\n\r");		/* go to new line */
		delay_ms(500);		/* wait 500ms */

	}
}

void keyboard_input(){
	if(keypad_kbhit() == 1){
			uint8_t key = keypad_getkey();
			switch(key){
				case 65:
						ok_pressed();
						break;
				case 66:
						up_pressed();
						break;
				case 67:
						down_pressed();
						break;
			}
		}
}

void set_current_temp(uint16_t u ){
	if(u != CURRENT_TEMP){
		CURRENT_TEMP = u;
	}
		lcd_gotoxy(3,2);
		char str[3];
		sprintf(str,"%d",u);
		lcd_print(str);
		lcd_gotoxy(14,2);
	//
	if(u >= TARGET_TEMP+3){
		first_cooler(1);
		second_cooler(1);
		first_heater(0);
		second_heater(0);
	}else if(u >= TARGET_TEMP+1){
		first_cooler(1);
		second_cooler(0);
		first_heater(0);
		second_heater(0);
	}else if(u <= (TARGET_TEMP-3)){
		first_cooler(0);
		second_cooler(0);
		first_heater(1);
		second_heater(1);
	}else if(u <= (TARGET_TEMP-1)){
		first_cooler(0);
		second_cooler(0);
		first_heater(1);
		second_heater(0);
	}
	if(u>MAX_TEMP || u< MIN_TEMP){
		buzzOn();
	}else{
		buzzOff();
	}
}


void ok_pressed(){
		buzzer();
		if(LAYER==1){
				LAYER =2;
				switch(COUNTER){
					case 0:
						show_temp_menu(TARGET_TEMP);
						break;
					case 1:
						show_temp_menu(MAX_TEMP);
						break;
					case 2:
						show_temp_menu(MIN_TEMP);
						break;
					case 3:
						break;
					case 4:
						break;
				}
		}else{
				LAYER = 1;
				show_menu();
		}
}
void up_pressed(){
		buzzer();
	if(LAYER==1){
			COUNTER ++;
			if(COUNTER+1>ARR_SIZE){
				COUNTER = 0;
			}
			show_menu();
	}else{
			//layer 2
		switch(COUNTER){
				case 0:
						//set temp
						TARGET_TEMP ++;
						show_temp_menu(TARGET_TEMP);
						break;
				case 1:
						//set max temp
						MAX_TEMP ++;
						show_temp_menu(MAX_TEMP);
						break;
				case 2:
						//set min temp
						MIN_TEMP ++;
						show_temp_menu(MIN_TEMP);
						break;
				case 3:
					//show logs
						break;
				case 4:
					//delete logs
						break;
		}
	}
}
void down_pressed(){
	if(LAYER==1){
			buzzer();
			COUNTER--;
			if(COUNTER<0){
				COUNTER = ARR_SIZE-1;
			}
			show_menu();
	}else{
			//layere 2
		switch(COUNTER){
				case 0:
						//set temp
						TARGET_TEMP --;
						show_temp_menu(TARGET_TEMP);
						break;
				case 1:
						//set max temp
						MAX_TEMP --;
						show_temp_menu(MAX_TEMP);
						break;
				case 2:
						//set min temp
						MIN_TEMP --;
						show_temp_menu(MIN_TEMP);
						break;
				case 3:
					//show logs
						break;
				case 4:
					//delete logs
						break;
		}
	}
}

void show_time(){
		lcd_gotoxy(7,2);
		char hour_str[2];
		char min_str[2];
		char sec_str[2];
		sprintf(hour_str,"%d",HOUR);
		sprintf(min_str,"%d",MIN);
		sprintf(sec_str,"%d",SEC);
		const char* hour= hour_str;
		const char* min= min_str;
		const char* sec= sec_str;
			strcpy(complete_hour,"");
		if(HOUR<10){
			strcat(complete_hour, "0");
		}
			strcat(complete_hour, hour);
			strcat(complete_hour, ":");
		if(MIN<10){
			strcat(complete_hour, "0");
		}
			strcat(complete_hour, min); 
			strcat(complete_hour, ":");
		if(SEC<10){
			strcat(complete_hour, "0");
		}
		strcat(complete_hour, sec);
		lcd_print(complete_hour);
		lcd_gotoxy(14,2);
}

void increase_sec(){
	SEC++;
	if(SEC>=60){
		SEC = 0;
		increase_min();
	}
}
void increase_min(){
	MIN++;
	if(MIN>=60){
		MIN = 0;
		increase_hour();
	}
}
void increase_hour(){
	HOUR++;
	if(HOUR>=24){
		HOUR = 0;
	}
}

void show_menu(){
		lcd_command(1);
		lcd_gotoxy(3,2);
		char str[3];
		sprintf(str,"%d",CURRENT_TEMP);
		lcd_print(str);
		//
		show_time();
		//
		lcd_gotoxy(1,1);
		char num[2];
		sprintf(num, "%d.", COUNTER+1);
		strcpy(menu_item,num);
		strcat(menu_item,MENU[COUNTER]);
		lcd_print(menu_item);
		lcd_gotoxy(14,2);
}

void show_temp_menu(uint16_t u){
		lcd_command(1);
		lcd_gotoxy(3,2);
		char str[3];
		sprintf(str,"%d",CURRENT_TEMP);
		lcd_print(str);
		//
		show_time();
		//
		lcd_gotoxy(1,1);
		char num[10];
		snprintf(num, 10, "temp = %d", u);
		lcd_print(num);
		lcd_gotoxy(14,2);
}
