codesg segment 'code'
		assume cs:codesg, ss:codesg, ds:codesg
		org 100h
	main:	jmp  prog1		
		a db 512 dup(?)
	prog1 proc near
		xor di,di
		mov bx,0100h
	lbl3:	mov al,[bx]
		cmp al,002eh
		jnz lbl
		mov di,0001h
	lbl:	cmp di,0001h
		jnz lbl2
		cmp al,61h
		jb  lbl2
		cmp al,7ah
		ja  lbl2
		sub al,20h
		mov [bx],al
	lbl2:	inc bx
		cmp bx,0200h
		jbe lbl3
		mov ax,4c00h
		int 21h
	prog1 endp
codesg ends
	end main