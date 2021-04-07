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

mess1 db "please enter x: ", 0
mess2 db "please enter y: ", 0
saida db "%lf",0
saida2 db "%lf power %lf is %lf",0

.data?

salvar1 dq ?
salvar2 dq ?
result dq ?

.code
start:
push offset mess1
call crt_printf
add esp,4
push offset salvar1
push offset saida
call crt_scanf
add esp,8
push offset mess2
call crt_printf
add esp,4
push offset salvar2
push offset saida
call crt_scanf
add esp,8

mov eax,dword ptr [salvar2+4]
push eax
mov eax,dword ptr [salvar2]
push eax
mov eax,dword ptr [salvar1+4]
push eax
mov eax,dword ptr [salvar1]
push eax
call crt_pow ;return in eax
add esp,16
fstp result



mov eax , dword ptr [result+4]
push eax
mov eax , dword ptr [result]
push eax
mov eax , dword ptr[salvar2+4]
push eax
mov eax , dword ptr[salvar2]
push eax
mov eax , dword ptr [salvar1+4]
push eax
mov eax , dword ptr [salvar1]
push eax
push offset saida2
call crt_printf
add esp,28

push 0
call ExitProcess

end start