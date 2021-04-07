BITS 64

SECTION .data

MyName:		db "Matin Noohnezhad",10
len_name:	equ $-MyName

SECTION .text

global _start

_start:
		mov rax,1
		mov rdi,1
		mov rsi,MyName
		mov rdx,len_name
		syscall
		
		mov rax,60
		mov rdi,0
		syscall