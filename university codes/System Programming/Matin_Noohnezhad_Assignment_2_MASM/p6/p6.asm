sseg segment 'stack'
	db 32 dup(?)
sseg ends
dseg segment 'data'
	a db 266 dup(?)
dseg ends
cseg segment 'code'
	main proc far	
		assume ss:sseg,cs:cseg,ds:dseg
		mov ax,dseg
		mov ds,ax
		mov bx,0109h
		mov al,[bx]
		dec bx
		mov ah,[bx]
		dec bx
		mov cl,[bx]
		dec bx
		mov ch,[bx]
		shl ax,1
		rcl cx,1
		mov bx,0109h
		mov [bx],al
		dec bx
		mov [bx],ah		
		dec bx
		mov [bx],cl		
		dec bx
		mov [bx],ch		
		mov bx,0105h
		mov al,[bx]
		dec bx
		mov ah,[bx]
		dec bx
		mov cl,[bx]
		dec bx
		mov ch,[bx]
		rcl ax,1
		rcl cx,1
		mov bx,0105h
		mov [bx],al
		dec bx
		mov [bx],ah		
		dec bx
		mov [bx],cl		
		dec bx
		mov [bx],ch		
		mov bx,0101h
		mov al,[bx]
		dec bx
		mov ah,[bx]
		rcl ax,1
		mov bx,0101h
		mov [bx],al
		dec bx
		mov [bx],ah
		mov ax,4c00h
		int 21h
	main endp
cseg ends
end main