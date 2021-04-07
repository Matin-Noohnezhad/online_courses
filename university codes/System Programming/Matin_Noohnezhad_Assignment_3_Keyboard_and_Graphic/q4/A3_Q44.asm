.model small
GRAPHIC_DATA segment at 0B800H
	pixels db 3f40h dup(?)
GRAPHIC_DATA ends
.code
		assume cs:@code
		;mov bx,@code
		;mov ds,bx
		org 100h
main: jmp load
oldint9 dd ?		;32-bit area to save the CS:IP of int 09
;----------- this portion remains resident
newint9 proc near
		push ax
		mov ah,2
		int 16h
		test al,00001000b
		jz over
		in al,60h
		cmp al,1fh
		jne over
		call starter
		;call starter
		;mov ah,0eh
		;mov al,07
		;int 10h
		int 20h
over:	pop ax
		assume ds:@code
		;mov bx, @code
		;mov ds,bx
		jmp cs:oldint9
newint9 endp
starter proc near
	assume ds:GRAPHIC_DATA
	mov bx,GRAPHIC_DATA
	mov ds,bx
	;mov ah,0
	;mov al,06
	;int 10h
	mov bx,0
lbl:
	mov di,3f3fh
	sub di,bx
	call reverser
	add bx,1
	cmp bx,1fa0h
	je finish
	jmp lbl
finish:	ret
starter endp
reverser proc near
	mov ch,0; ch is complete reverse of [bx]
	mov al,[bx]
	mov ah,al
	mov cl,1
	rol al,cl
	and al,1
	or ch,al
	mov al,ah
	mov cl,3
	rol al,cl
	and al,2
	or ch,al
	mov al,ah
	mov cl,3
	ror al,cl
	and al,4
	or ch,al		
	mov al,ah
	mov cl,1
	ror al,cl
	and al,8
	or ch,al	
	mov al,ah
	mov cl,1
	rol al,cl
	and al,10h
	or ch,al
	mov al,ah
	mov cl,3
	rol al,cl
	and al,20h
	or ch,al
	mov al,ah
	mov cl,3
	ror al,cl
	and al,40h
	or ch,al
	mov al,ah
	mov cl,1
	ror al,cl
	and al,80h
	or ch,al; ch is ready

	mov dh,0; dh is complete reverse of [di]
	mov al,[di]
	mov ah,al
	mov cl,1
	rol al,cl
	and al,1
	or dh,al
	mov al,ah
	mov cl,3
	rol al,cl
	and al,2
	or dh,al
	mov al,ah
	mov cl,3
	ror al,cl
	and al,4
	or dh,al		
	mov al,ah
	mov cl,1
	ror al,cl
	and al,8
	or dh,al	
	mov al,ah
	mov cl,1
	rol al,cl
	and al,10h
	or dh,al
	mov al,ah
	mov cl,3
	rol al,cl
	and al,20h
	or dh,al
	mov al,ah
	mov cl,3
	ror al,cl
	and al,40h
	or dh,al
	mov al,ah
	mov cl,1
	ror al,cl
	and al,80h
	or dh,al; dh is ready
	mov [di],ch
	mov [bx],dh
	ret
reverser endp
;-------------- this portion is run once only during the initialization
		assume ds:@code
		;mov bx,@code
		;mov dx,bx
load proc near
		mov ah,35h
		mov al,09h
		int 21h
		mov word ptr oldint9,bx
		mov word ptr oldint9+2,es
		mov ah,25h
		mov al,09h
		mov dx,offset newint9
		int 21h
		mov dx,(offset load - offset @code)
		add dx,15
		mov cl,4
		shr dx,cl
		mov ah,31h
		int 21h
load endp
	end main


