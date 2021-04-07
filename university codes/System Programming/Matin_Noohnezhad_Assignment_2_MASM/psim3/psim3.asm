.model compact
.stack 32
.data
	nouse db 256 dup(?)
	a db 256 dup(?)	

.data?
	nouse2 db 256 dup(?)
	nouse3 db 256 dup(?)
	nouse4 db 256 dup(?)
	b db 32 dup(?)
.code
	main proc FAR
		mov di,00ffh
		mov cx,0100h
		mov dx,0300h
	start:	mov bx,cx
		;assume ds:dataseg1, ss:stackseg, cs:codeseg
		mov ax,@data
		mov ds,ax
		mov al,[bx]
		;assume ds:dataseg2, ss:stackseg, cs:codeseg
		;mov ax,@const
		;mov ds,ax
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