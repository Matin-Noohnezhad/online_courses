.model small
graphic_data segment at 0b800h
	pix_even db 8000 dup(?)
	nothing db 192 dup(?)
	pix_odd db 8000 dup(?)
graphic_data ends
.stack 64
.data
x1 dw 200; x1<=639
y1 db 60; ; y1<=199
x2 dw 50; x2<=639
y2 db 50; y2<=199
.code
main proc
	push ax
	push bx
	push cx
	push dx
	push si
	assume ds:@data
	mov bx, @data
	mov ds,bx
	mov ax,6
	int 10h
	;
	call y_chg_coord; y = 199 - y
	call x_sort; if(x1>x2) then swap((x1,y1),(x2,y2))
	call x_diff; ax = x2-x1
	mov cx,ax
	call y_sort; if(y1>y2) then swap((x1,y1),(x2,y2))
	call y_diff; al = y2-y1
	xor ah,ah; ax = y2-y1
	;
	cmp cx,ax
	jb lbl_
	;x_diff > y_diff
	call x_inc
	jmp fin
lbl_:; y_diff > x_diff
	call y_inc
	;	
fin:
	pop si
	pop dx
	pop cx
	pop bx
	pop ax
	int 20h
main endp
x_inc proc
	push ax
	push bx
	push cx
	push dx
	push si
	call grad_status
	cmp al,2
	je g0
	cmp al,1
	je gp
	jmp gn
g0:
;gradient of line = 0
	call x_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
	mov bx,offset x2
	mov ax,[bx]
lbl_xi:
	call put_pixel
	add cx,1
	cmp cx,ax
	jbe lbl_xi
	jmp x_inc_fin
gp:
;gradient of line > 0
	call x_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
lbl_gp:	
	call put_pixel
	assume ds:@data
	mov bx, @data
	mov ds,bx
	add cx,1
	mov bx,offset x2
	cmp cx,[bx]
	ja x_inc_fin
	mov bx, offset x1
	mov ax,[bx]
	mov bx,cx
	sub bx,ax; bx = x-x0
	mov si,offset y2
	mov dl,[si]
	mov si,offset y1
	mov dh,[si]
	sub dl,dh; dl = y2-y1
	mov ax,bx
	xor dh,dh
	mul dx; dxax <- (y2-y1)*(x-x0)
	mov bx,offset x2
	push cx
	mov cx,[bx]
	mov bx,offset x1
	mov bx,[bx]
	sub cx,bx
	div cx; quotient in ax --- [(y2-y1)/(x2-x1)]*(x-x0)
	pop cx ; x(new) = cx
	mov bx,offset y1
	mov dl,[bx]
	xor dh,dh
	add ax,dx
	mov dx,ax; y(new) = dx
	jmp lbl_gp
	
gn:
;gradient of line < 0
	call x_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
lbl_gn:	
	call put_pixel
	assume ds:@data
	mov bx, @data
	mov ds,bx
	add cx,1
	mov bx,offset x2
	cmp cx,[bx]
	ja x_inc_fin
	mov bx, offset x1
	mov ax,[bx]
	mov bx,cx
	sub bx,ax; bx = x-x0
	mov si,offset y2
	mov dh,[si]
	mov si,offset y1
	mov dl,[si]
	sub dl,dh; dl = y1-y2
	mov ax,bx
	xor dh,dh
	mul dx; dxax <- (y1-y2)*(x-x0)
	mov bx,offset x2
	push cx
	mov cx,[bx]
	mov bx,offset x1
	mov bx,[bx]
	sub cx,bx
	div cx; quotient in ax --- [(y1-y2)/(x2-x1)]*(x-x0)
	pop cx ; x(new) = cx
	mov bx,offset y1
	mov dl,[bx]
	xor dh,dh
	sub dx,ax; y(new) = dx
	jmp lbl_gn
x_inc_fin:
	pop si
	pop dx
	pop cx
	pop bx
	pop ax
	ret	
x_inc endp
y_inc proc
	push ax
	push bx
	push cx
	push dx
	push si
	call grad_status
	cmp al,3
	je g_inf
	cmp al,1
	je gp2
	jmp gn2
g_inf:
;gradient of line = 0
	call y_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
	mov bx,offset y2
	mov al,[bx]
	xor ah,ah
lbl_xi2:
	call put_pixel
	add dx,1
	cmp dx,ax
	jbe lbl_xi2
	jmp y_inc_fin
gp2:
;gradient of line > 0
	call y_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
lbl_gp2:	
	call put_pixel
	assume ds:@data
	mov bx, @data
	mov ds,bx
	add dx,1
	push dx
	mov bx,offset y2
	cmp dl,[bx]
	ja y_inc_fin
	mov bx,offset y1
	mov cl,[bx]
	xor ch,ch
	sub dx, cx; dx = y-y0
	mov bx, offset x2
	mov ax,[bx]
	mov bx, offset x1
	mov bx,[bx]
	sub ax,bx; ax = x2-x1
	mul dx; dxax = (x2-x1)*(y-y0)
	mov bx,offset y2
	mov cl,[bx]
	mov bx,offset y1
	mov ch,[bx]
	sub cl,ch
	xor ch,ch; cx = y2-y1
	div cx; quotient in ax --- [(x2-x1)/(y2-y1)]*(y-y0)
	mov bx, offset x1
	mov bx,[bx]
	add ax,bx
	mov cx,ax; x(new) = cx
	pop dx; y(new) = dx
	jmp lbl_gp2
	
gn2:
;gradient of line < 0
	call y_sort
	mov bx,offset x1
	mov si,offset y1
	mov cx,[bx]
	xor dh,dh
	mov dl,[si]
lbl_gn2:	
	call put_pixel
	assume ds:@data
	mov bx, @data
	mov ds,bx
	add dx,1
	push dx
	mov bx,offset y2
	cmp dl,[bx]
	ja y_inc_fin
	mov bx,offset y1
	mov cl,[bx]
	xor ch,ch
	sub dx, cx; dx = y-y0
	mov bx, offset x1
	mov ax,[bx]
	mov bx, offset x2
	mov bx,[bx]
	sub ax,bx; ax = x1-x2
	mul dx; dxax = (x1-x2)*(y-y0)
	mov bx,offset y2
	mov cl,[bx]
	mov bx,offset y1
	mov ch,[bx]
	sub cl,ch
	xor ch,ch; cx = y2-y1
	div cx; quotient in ax --- [(x2-x1)/(y2-y1)]*(y-y0)
	mov bx, offset x1
	mov bx,[bx]
	sub bx,ax
	mov cx,bx; x(new) = cx
	pop dx; y(new) = dx
	jmp lbl_gn2
y_inc_fin:
	pop si
	pop dx
	pop cx
	pop bx
	pop ax
	ret	
y_inc endp
grad_status proc
;(if gradient of line <0 then al=0)
;(if gradient of line >0 then al=1)
;(if gradient of line = 0 then al=2)
;(if gradient of line = inf then al=3)
	push bx
	mov bx, offset x1
	mov ax, [bx]
	mov bx, offset x2
	mov bx, [bx]
	cmp ax,bx
	jnz lbl_g
	mov al,3
	jmp fin_g
lbl_g:
	mov bx, offset y1
	mov al, [bx]
	mov bx, offset y2
	mov bl, [bx]
	cmp al,bl
	jnz lbl_g2
	mov al,2
	jmp fin_g	
lbl_g2:
	mov bx, offset x1
	mov ax, [bx]
	mov bx, offset x2
	mov bx, [bx]
	cmp ax,bx
	jb lbl_g3
	;x1>x2
	mov bx, offset y1
	mov al, [bx]
	mov bx, offset y2
	mov bl, [bx]
	cmp al,bl
	ja lbl_g5
	;y1<y2
	mov al,0
	jmp fin_g	
lbl_g5:
	;y1>y2
	mov al,1
	jmp fin_g
lbl_g3:
	;x1<x2
	mov bx, offset y1
	mov al, [bx]
	mov bx, offset y2
	mov bl, [bx]
	cmp al,bl
	jb lbl_g4
	;y1>y2
	mov al,0
	jmp fin_g
lbl_g4:
	;y1<y2
	mov al,1
fin_g:	
	pop bx
	ret
grad_status endp
x_diff proc; ax is output
	push bx
	mov bx, offset x2
	mov ax, [bx]
	mov bx, offset x1
	mov bx, [bx]
	sub ax,bx
	pop bx
	ret
x_diff endp
y_diff proc; al is output
	push bx
	mov bx, offset y2
	mov al, [bx]
	mov bx, offset y1
	mov ah, [bx]
	sub al,ah
	pop bx
	ret
y_diff endp
x_sort proc
	push ax
	push bx
	push dx
	push si
	mov bx, offset x1
	mov ax,[bx]
	mov si, offset x2
	mov dx,[si]
	cmp ax,dx
	jna fin1
	mov [bx],dx
	mov [si],ax
	mov bx, offset y1
	mov al,[bx]
	mov si, offset y2
	mov ah,[si]
	mov [bx],ah
	mov [si],al
fin1:
	pop si
	pop dx
	pop bx
	pop ax
	ret
x_sort endp
y_sort proc
	push ax
	push bx
	push dx
	push si
	mov bx, offset y1
	mov al,[bx]
	mov si, offset y2
	mov dl,[si]
	cmp al,dl
	jna fin2
	mov [bx],dl
	mov [si],al
	mov bx, offset x1
	mov ax,[bx]
	mov si, offset x2
	mov dx,[si]
	mov [bx],dx
	mov [si],ax
fin2:
	pop si
	pop dx
	pop bx
	pop ax
	ret
y_sort endp
y_chg_coord proc
	push ax
	push bx
	mov bx,offset y1
	mov al,199
	mov ah,byte ptr [bx]
	sub al,ah
	mov byte ptr [bx],al
	mov bx,offset y2
	mov al,199
	mov ah,byte ptr [bx]
	sub al,ah
	mov byte ptr [bx],al
	pop bx
	pop ax
	ret
y_chg_coord endp
put_pixel proc; input cx=x , dx=y
	push bx
	push ax
	push cx
	push dx
	;;;;
	assume ds:graphic_data
	mov bx, graphic_data
	mov ds,bx
	;;;;
	mov bx,dx
	and bx,1
	cmp bx,1
	je lbl
	;dx is even
	shr dx,1
	;we must calculate dx*640+cx  ==>640 = 2^7 * 5
	push cx
	mov cl,7
	shl dx,cl ; dx = dx *2^7
	pop cx
	mov bx, dx; dx *2^7
	push cx
	mov cl ,2
	shl dx,cl; 4*(dx *2^7)
	pop cx
	add dx,bx; dx = 5*(dx *2^7)
	add dx , cx ; dx = dx*640+cx
	mov ax,dx
	mov dx, ax
	and dx,00000111b;remainder in dx
	push cx
	mov cl,3
	shr ax , cl; quotient in ax
	pop cx
	mov bx,offset pix_even
	jmp last_lbl
	
	
lbl:;dx is odd
	sub dx,1
	shr dx,1
	;we must calculate dx*640+cx  ==>640 = 2^7 * 5
	push cx
	mov cl,7
	shl dx,cl ; dx = dx *2^7
	pop cx
	mov bx, dx; dx *2^7
	push cx
	mov cl , 2
	shl dx,cl; 4*(dx *2^7)
	pop cx
	add dx,bx; dx = 5*(dx *2^7)
	add dx , cx ; dx = dx*640+cx
	mov ax,dx
	mov dx,ax
	and dx, 00000111b ; remainder in dx
	push cx
	mov cl,3
	shr ax,cl; quotient in ax
	pop cx
	mov bx,offset pix_odd
last_lbl:	
	add bx,ax
	cmp dx,0
	je lbl0
	cmp dx,1
	je lbl1
	cmp dx,2
	je lbl2
	cmp dx,3
	je lbl3
	cmp dx,4
	je lbl4
	cmp dx,5
	je lbl5
	cmp dx,6
	je lbl6
	cmp dx,7
	je lbl7
lbl0:
	mov al, 10000000b
	or [bx] , al
	jmp finish
lbl1:
	mov al, 01000000b
	or [bx] , al
	jmp finish
lbl2:
	mov al, 00100000b
	or [bx] , al
	jmp finish
lbl3:
	mov al, 00010000b
	or [bx] , al
	jmp finish
lbl4:
	mov al, 00001000b
	or [bx] , al
	jmp finish
lbl5:
	mov al, 00000100b
	or [bx] , al
	jmp finish
lbl6:
	mov al, 00000010b
	or [bx] , al
	jmp finish
lbl7:	
	mov al, 00000001b
	or [bx] , al	
	jmp finish
	;;;;;;
finish:
	pop dx
	pop cx
	pop ax
	pop bx
	ret
put_pixel endp
end main