.model small
.code
	org 100h
	main:	jmp  prog1		
		n1 dw ?
		n2 db ?
		n3 dw ?
		n4 dw ?			
	prog1 proc near
		xor ah,ah
		mov si,ax
		shl si,1
		add si,ax
		mov n2,al
		mul ax
		mov n1,ax
		shl ax,1
		mov bx,ax
		sub bx,si
		add bx,6
		xor di,di
		shl ax,1
		rcl di,1
		mov cx,ax
		mov si,di
		mov dl,n2
		xor dh,dh
		mul dx
		mov n3,dx
		mov n4,ax
		mov al,n2
		xor ah,ah
		mul di
		add n3,ax
		mov ax,n1
		add ax,cx
		jnc lbl
		add si,1
	  lbl:  mov di,n3
		mov cx,n4
		sub cx,ax
		sbb di,si
		mov ax,000ch
		xor si,si
		sub cx,ax
		sbb di,si
		mov dx,di
		mov ax,cx
		div bx
		mov bx,ax
		mov ax,4c00h
		int 21h
	prog1 endp
end main