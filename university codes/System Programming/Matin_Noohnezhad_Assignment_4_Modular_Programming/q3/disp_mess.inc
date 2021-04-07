disp_mess macro message
local text
local lbl
jmp lbl
text db message,0
lbl:
push offset text
call crt_printf
add esp , 4
endm