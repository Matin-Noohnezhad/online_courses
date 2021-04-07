.model small
GRAPHIC_DATA segment at 0B800H
	pixels db 3f40h dup(?)
GRAPHIC_DATA ends
.code
	org 100h
main proc near
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
finish:	int 20h
main endp
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
	end main