disp_n macro mess,n
local text
local lbl
local lbl2
jmp lbl
text db mess,0
lbl:
mov ecx,n
lbl2:
push ecx
push offset text
call crt_printf
add esp,4
pop ecx
loop lbl2
endm