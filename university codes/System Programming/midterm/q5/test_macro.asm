.386
.model flat,c
option casemap:none
include \masm32\include\windows.inc
include \masm32\include\masm32.inc
include \masm32\include\msvcrt.inc
include \masm32\include\kernel32.inc
include macro.inc

includelib \masm32\lib\masm32.lib
includelib \masm32\lib\msvcrt.lib
includelib \masm32\lib\kernel32.lib

.code
main proc near
disp_n "Matin, ",10
disp_n "Noohnezhad, ",10
push 0
call ExitProcess
main endp
end main

