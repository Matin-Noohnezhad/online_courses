codeseg segment
		assume cs:codeseg
		org 100h
main: jmp short load
oldint10 dd ?		;32-bit area to save the CS:IP of int 09
;----------- this portion remains resident
newint10 proc
		cmp ah,9
		je lbl
		cmp ah,0ah
		je lbl
		jmp cs:oldint10
lbl:
		mov al,78h
		jmp cs:oldint10
newint10 endp
;-------------- this portion is run once only during the initialization
		assume cs:codeseg,ds:codeseg
load proc near
		mov ah,35h
		mov al,10h
		int 21h
		mov word ptr oldint10,bx
		mov word ptr oldint10+2,es
		mov ah,25h
		mov al,10h
		mov dx,offset newint10
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


