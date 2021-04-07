#include <stm32f10x.h>

extern void lcd_init(void);
extern void lcd_command(unsigned char cmd);
extern void lcd_data(unsigned char data);
extern void lcd_gotoxy(unsigned char x, unsigned char y);
extern void lcd_print(char * str);
