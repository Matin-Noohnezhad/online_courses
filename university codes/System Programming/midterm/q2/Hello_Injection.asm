inject macro input
	local character
	local lbl
	local lbl2
	local finish
	jmp lbl
	character db input
lbl:
	mov bx,Head
	add bx,2
	cmp bx,offset label
	jne lbl2	
	mov bx, offset buffer
lbl2:
	cmp bx,Tail
	je finish
	mov si, offset character
	mov [bx],byte ptr cs:[si]
	mov Head, bx
finish:
endm

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
	
	CLI
	inject 'H'
	inject 'E'
	inject 'L'
	inject 'L'
	inject 'O'
	STI
	
	int 20h
main endp
end main




