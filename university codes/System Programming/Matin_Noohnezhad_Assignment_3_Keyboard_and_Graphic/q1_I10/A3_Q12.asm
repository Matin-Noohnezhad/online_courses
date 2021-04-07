.model small
ROM_BIOS_DATA segment at 40H
	org 1AH
	Head dw ?
	Tail dw ?
	buffer dw 16 dup (?)
	buffer_end  label word
ROM_BIOS_DATA ends

.code
	org 100h
main proc far
	assume ds:ROM_BIOS_DATA
	mov bx, ROM_BIOS_DATA
	mov ds,bx
	mov bx,Head
	cmp bx, 3Ch
	jne lbl
	mov bx, 1Eh
lbl:	add bx,2
lbl2:	cmp bx, Head
	je finish
	mov ah,0Ah
	mov al, [bx]
	mov cx,1
	int 10h
	push bx
	mov ah,03h
	mov bh,0
	int 10h
	mov ah,02h
	add dl,1
	int 10h
	pop bx
	add bx,2
	cmp bx,3Ch
	jna lbl3
	mov bx,1Eh
lbl3: jmp lbl2
finish:	int 20h
main endp
end main




