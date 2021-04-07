.386                            ; Tells MASM to use Intel 80386 instruction set.

.model flat,stdcall             ; Flat memory model

option casemap:none             ; Treat labels as case-sensitive

include \masm32\include\windows.inc
include \masm32\include\masm32.inc
include \masm32\include\msvcrt.inc
include \masm32\include\kernel32.inc
include disp_mess.inc

includelib \masm32\lib\masm32.lib
includelib \masm32\lib\msvcrt.lib
includelib \masm32\lib\kernel32.lib

.code
start proc near
disp_mess "My fname is matin."
disp_mess " My lname is noohnezhad."
push 0
call ExitProcess
start endp
end
