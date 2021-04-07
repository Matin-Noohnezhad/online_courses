.model small
.stack 64
.data
	nouse db 256 dup(?)
	a db 256 dup(?)
	nouse2 db 256 dup(?)
	b db 32 dup(?) 	
.code
	main proc far
		mov ax,@data
		mov ds,ax
		mov di,00ffh
		mov cx,0100h
		mov dx,0300h
	start:	mov bx,cx
		mov al,[bx]
		mov bx,dx
		cmp al,[bx]
		jnz lbl
		inc dx
		cmp di,00ffh
		jz lbl3
		jmp lbl2
	lbl3:	mov di,cx
		sub di,0100h
		jmp lbl2
	lbl:	mov dx,300h
		mov di,00ffh
	lbl2:	inc cx
		cmp dx,0321h
		jz end1
		cmp cx,0201h
		jz end1
		jmp start
	end1:	mov ax,di
		mov ax,4c00h
		int 21h
	main endp
end main