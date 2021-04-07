.model small
.stack 64
.data
	str1 db 'enter first string...','$'
	size_of_str2 db 30
	str2 db 30 dup(0)
	str3 db 'enter second string...','$'
	size_of_str4 db 30
	str4 db 30 dup(0)
	disp1 db 'They are equal','$'
	disp2 db 'They are not equal','$'
.code
main:
	assume ds:@data , es:@data
	mov ax,@data
	mov ds,ax
	mov es,ax
	mov dx, offset str1
	mov ah,09h
	int 21h
	mov dx, offset size_of_str2
	mov ah,0ah
	int 21h
	mov di,offset str2
	xor bh,bh
	mov bl,[di]
	mov [di+bx+1],'$'
	;;
	mov dx, offset str3
	mov ah,09h
	int 21h
	mov dx, offset size_of_str4
	mov ah,0ah
	int 21h
	mov di,offset str4
	xor bh,bh
	mov bl,[di]
	mov [di+bx+1],'$'
	;;
	mov cx,30
	mov si,offset str2
	add si,1
	mov di,offset str4
	add di,1
	repe cmpsb
	je lbl
	mov dx, offset disp2
	mov ah,09h
	int 21h
	jmp lbl2
	;;
lbl:	mov dx, offset disp1
	mov ah,09h
	int 21h
lbl2:	mov ax, 4c00h
	int 21h
end main