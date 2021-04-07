init_data macro message
	local text
	text db message , '$'
endm
disp_mess macro message
	mov ah,09
	mov dx, offset message
	int 21h
endm
.model small
.stack 64
.data
init_data 'my name '
init_data 'is '
init_data 'matin.'
;mess1 db 'my name ','$'
;mess2 db 'is ' , '$'
;mess3 db 'matin' , '$'
.code
main proc far
mov ax,@data
mov ds,ax
disp_mess ??0000
disp_mess ??0001
disp_mess ??0002
mov ax, 4c00h
int 21h
main endp
end main
