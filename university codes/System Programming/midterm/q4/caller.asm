.386
.model flat,c

option casemap:none 
include \masm32\include\windows.inc
include \masm32\include\masm32.inc
include \masm32\include\msvcrt.inc
include \masm32\include\kernel32.inc

includelib \masm32\lib\masm32.lib
includelib \masm32\lib\msvcrt.lib
includelib \masm32\lib\kernel32.lib

extern func:near

.data
a dd 3
b dd 2
result dd ?
mess db 'the result is %d',0

.code
main proc
mov eax,b
push eax
mov eax,a
push eax
call func
mov result,eax
push result
push offset mess
call crt_printf
invoke ExitProcess,0 
main endp
end main