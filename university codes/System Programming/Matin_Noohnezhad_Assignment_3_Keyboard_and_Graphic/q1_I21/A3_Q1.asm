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
	mov ah,02
	mov dl, [bx]
	int 21h
	add bx,2
	cmp bx,3Ch
	jna lbl3
	mov bx,1Eh
lbl3: jmp lbl2
finish:	int 20h
main endp
end main




