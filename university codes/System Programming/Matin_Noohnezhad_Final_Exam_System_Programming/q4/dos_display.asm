.model small
.stack 32
.data
	msg db 'Matin Noohnezhad$'
.code
	main:
		mov ax, @data
		mov ds, ax
		mov ah, 09h
		mov dx,offset msg
		int 21h
		mov ax, 4c00h
		int 21h		
end main