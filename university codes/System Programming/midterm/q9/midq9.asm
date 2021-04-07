ROM_BIOS_DATA segment at 40H
	org 1AH
	Head dw ?
	Tail dw ?
	buffer dw 16 dup (?)
	buffer_end  label word
ROM_BIOS_DATA ends

codeseg segment
		assume cs:codeseg
		org 100h
main: jmp short load
oldint9 dd ?		;32-bit area to save the CS:IP of int 09
;----------- this portion remains resident
newint9 proc
		CLI
		call cs:oldint9
		assume ds:ROM_BIOS_DATA
		mov bx,ROM_BIOS_DATA
		mov ds,bx
		mov bx,Tail
		mov [bx],78h
		iret
		STI
newint9 endp
;-------------- this portion is run once only during the initialization
		assume cs:codeseg,ds:codeseg
load proc near
		mov ah,35h
		mov al,9h
		int 21h
		mov word ptr oldint9,bx
		mov word ptr oldint9+2,es
		mov ah,25h
		mov al,9h
		mov dx,offset newint9
		int 21h
		mov dx,(offset load - offset codeseg)
		add dx,15
		mov cl,4
		shr dx,cl
		mov ah,31h
		int 21h
load endp
codeseg ends
	end main


