.386
.model flat,stdcall

option casemap:none
include \masm32\include\windows.inc
include \masm32\include\masm32.inc
include \masm32\include\msvcrt.inc
include \masm32\include\kernel32.inc

includelib \masm32\lib\masm32.lib
includelib \masm32\lib\msvcrt.lib
includelib \masm32\lib\kernel32.lib

.data

mess1 db "please enter a: ", 0
mess2 db "please enter b: ", 0
mess3 db "they are equal.",0
mess4 db "they are not equal",0
saida db "%s",0


.data?

str1 db 30 dup(?)
str2 db 30 dup(?)

.code
main:
push  offset mess1
call crt_printf
add esp,4
push offset str1
push offset saida
call crt_scanf
add esp,8
push  offset mess2
call crt_printf
add esp,4
push offset str2
push offset saida
call crt_scanf
add esp,8

push offset str2
push offset str1
call crt_strcmp;return is in eax
add esp,8

cmp eax,0
je lbl
push  offset mess4
call crt_printf
add esp,4
jmp lbl2
lbl:
push  offset mess3
call crt_printf
add esp,4
lbl2:
jmp lbl2;just for pooling so you can see the answer
push 0
call ExitProcess

end main