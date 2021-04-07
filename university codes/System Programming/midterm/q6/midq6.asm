.model small
graphic_buffer segment at 0b800h
shapes db 4000 dup(?)
lbl db ?
graphic_buffer ends

.code
main proc far
assume ds:graphic_buffer,es:graphic_buffer
mov bx,graphic_buffer
mov ds,bx
mov Es,bx
;+ = 2B , color = 7a
mov cx,2000
mov ax,7a2bh
mov di, offset shapes
cld
rep stosw
mov ax,4c00h
int 21h
main endp
end main



