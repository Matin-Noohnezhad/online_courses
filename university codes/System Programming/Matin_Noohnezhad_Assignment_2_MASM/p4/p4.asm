sseg segment 'stack'
	db 32 dup(?)
sseg ends
dseg segment 'data'
	a dw 0
	b dw 5
	cc dw 0
	d dw 3
	e dw ?
	f dw ?
	g dw ?
	h dw ?
	i dw ?
	j dw ?
dseg ends
cseg1 segment 'code'
	main proc far
		assume cs:cseg1,ds:dseg,ss:sseg
		mov ax,dseg
		mov ds,ax
		mov ax,a
		mov bx,b
		mov cx,cc
		mov dx,d
		call multiplication
		mov ax,4c00h
		int 21h	
	main endp
cseg1 ends
cseg2 segment 'code'
	multiplication proc far
		assume cs:cseg2
		mov di,dx
		mov si,ax
		mov ax,bx
		mul di
		mov g,ax
		mov a , dx
		mov ax,si
		mul di
		mov b, ax
		mov d,dx
		mov ax,bx
		mul cx
		mov cc,ax
		mov cc,dx
		mov ax,si
		mul cx
		mov j,dx
		mov f,ax
		xor si,si
		xor di,di
		mov ax,a
		add ax,b
		jnc lbl
		add di,1
	lbl:add ax,cc
		jnc label2
		add di,1
	label2: mov h,ax
		mov bx,d
		add bx,cc
		jnc  label3
		add si,1
	label3:add bx,f
		jnc label4
		add si,1
	label4: add bx,di
		jnc label5
		add si,1
	label5: mov i,bx
		add j,si
		mov ax,j
		mov bx,i
		mov cx,h
		mov dx,g
		ret
	multiplication endp
cseg2 ends
end main