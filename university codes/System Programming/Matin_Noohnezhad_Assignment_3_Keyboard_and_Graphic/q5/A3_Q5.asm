.model small

keyboard_buffer segment at 40h
	org 1ah
	head dw ?
	tail dw ?
	buffer dw 16 dup(?)
	buffer_end label word
keyboard_buffer ends		
.code
		assume cs:@code
		org 100h
main: jmp short load
oldint9 dd ?		;32-bit area to save the CS:IP of int 09
;----------- this portion remains resident
newint9 proc
		push ax
		mov ah,0
		int 16h
		cmp al,41h
		jb over
		cmp al,5Ah
		ja over
		add al,20h
		assume ds:keyboard_buffer	
		mov bx, keyboard_buffer
		mov ds,bx
		mov bx,head
		add bx,2
		cmp bx,3ch
		ja lbl		
		mov [bx],ah
		jmp lbl2
lbl:		mov bx,1eh
		mov [bx], ah	
lbl2:		mov head,bx
		;int 20h
		iret
		;test al,00001000b
		;jz over
		;in al,60h
		;cmp al,1fh
		;jne over
		
		;mov ah,0eh
		;mov al,07
		;int 10h
over:	pop ax
		jmp cs:oldint9
newint9 endp
;-------------- this portion is run once only during the initialization
		assume cs:@code,ds:@code
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


