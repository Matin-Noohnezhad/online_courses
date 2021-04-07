.model small

.stack 64

.data
		r db 6 dup(0)
		s db 10 dup(?)
		zero db 0
		d db 5 dup(?)

.code 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	main proc near
		mov cx , 80
		mov ax , @data
		mov ds,ax
lbl:
		mov bx , offset r
		push cx
		mov cx,16
		mov dx , 1
		call shift_left ;(r , 16byte , 1bit ) - bx = r , cx = 16 , dx = 1
		_cmp (r , zero , 6)
		jb lbl2
		_sub(r , zero , 5)
		or [s+9] , 00000001b
		pop cx
lbl2:	loop lbl
		int 20h
		
	main endp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	shift_left proc near ;(r , 16byte , 1bit ) - bx = r , cx = 16 , dx = 1
lbl4:
		dec dx
		clc ; clear flag
		pushf 
lbl3:	
		popf
		rcl [bx+cx-1] , 1
		pushf
		loop lbl3
		cmp dx , 0
		ja lbl4
		ret
	shift_left endp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	_sub proc near
	;...
	_sub endp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	_cmp proc near
	;...
	_cmp endp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	
	
end main

