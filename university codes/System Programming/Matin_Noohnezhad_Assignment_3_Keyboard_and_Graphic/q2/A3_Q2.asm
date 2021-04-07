.model small
GRAPHIC_DATA segment at 0B800H
	pixels dw 2000 dup(?)
GRAPHIC_DATA ends
.code
	org 100h
main proc near
	assume ds:GRAPHIC_DATA
	mov bx,GRAPHIC_DATA
	mov ds,bx
	mov bx,0
lbl:	mov cl,[bx+1]	
	mov al,0ffh
	sub al,cl
	mov [bx+1],al
	add bx,2
	cmp bx,4000
	je finish
	jmp lbl
finish:	int 20h
main endp
	end main