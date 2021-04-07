.model small 
.stack 64
.data
	cmd_len db ?
	cmdl db 127 dup(?)
.code
start:
	mov dx, @data
	mov ds, dx
	;get psp address
	mov ah, 51h
	int 21h; psp address is in bx
	mov es, bx; psp address is in es 	
	;set cmdl and cmdl_length
	mov ah, [es:80h]
	mov cmd_len, ah
	;
	mov di, offset cmdl
	mov si, 81h
	;swap ds and es
	mov ax, es
	mov bx, ds
	mov es, bx
	mov ds, ax
	;
	mov cx, 127
	rep movsb ; mov from es:si to ds:di
	;swap ds and es
	mov ax, es
	mov bx, ds
	mov es, bx
	mov ds, ax
	;
	;put $ at the end of the cmdl
	mov bx, offset cmdl
	xor ax,ax
	mov al, cmd_len
	add bx, ax
	mov [bx], "$"
	;print string
	mov dx,offset cmdl
	mov ah, 09h
	int 21h
	;terminate program
	mov ax,4c00h
	int 21h

end start