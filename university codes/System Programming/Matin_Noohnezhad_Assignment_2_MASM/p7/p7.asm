sseg segment 'stack'
	db 32 dup(?)
sseg ends
dseg1 segment 'data'
	a db 10 dup(?)
	temp1 dw ?


dseg1 ends
dseg2 segment 'data'
	b db 10 dup(?)
	temp2 dw ?
dseg2 ends
cseg1 segment 'code'
	main proc far
		assume cs:cseg1,ds:dseg1,ss:sseg
		mov ax,dseg1
		mov ds,ax		
		call shifter
		mov ax,4c00h
		int 21h
	main endp
cseg1 ends
cseg2 segment 'code'
	shifter proc far
		xor di,di
		xor bx,bx
		mov cx,5
	lbl:	mov si,cx
		push bx
		mov bx,9
		sub bx,di
		mov al,[bx]
		mov ah,[bx-1]
		mov dl,ah
		mov cl,4
		shl ax,cl
		mov [temp1],bx
		pop bx
		or al,bl
		mov bx,temp1
		mov [bx],al
		mov [bx-1],ah
		mov bl,dl
		shr bl,cl
		add di,2
		mov cx,si
		loop lbl
		ret
	shifter endp
cseg2 ends
end main



