.model small

.stack 64

.data
	lpt2PinNo 			dw  0378h ; 03BC --> data, 03BD --> status, 03BE --> control 
	nibbleFirstPart 	db	1 ; Is it first part of nibble or second part of it.
	DataOffset			dw	0 ; Which offset of data are we sending right now.
.data?
	dataToReceive 		db  1000 dup(?) ; data to receive
	
.code
	start: 
		mov		dx, @data
		mov		ds, dx
		mov 	dx, lpt2PinNo
		inc 	dx
		inc 	dx  ;control port
		in		al, dx
		or		al, 00100000b 
		out		dx, al	; make it backward direction by set bit 5 of control port to 1
		;still control port for checking strobe bit
	check_finish:
		cmp DataOffset, 1000
		jz  finish
	strobe_check:
		in  	al, dx
		test	al, 00000001b
		jnz		strobe_check	; if strobe is 1 check again until it became 0
		;strobe is 0
		or 		al, 00000001b
		out		dx, al ; set strobe to 1
		dec 	dx  ;status port 
		in 		al, dx
		or		al, 10000000b
		out		dx, al ; set busy to 1
		dec		dx	;data port		
		mov 	bl, nibbleFirstPart
		test	bl, 1
		jz		nibble_second_part
		;nibble first part
		in 	al, dx
		lea bx, dataToReceive
		mov	si, DataOffset
		mov cx, 4
		shl al, cx; now the nibble is in the most 4 significant bit
		mov [bx][si], al
		xor bl, bl
		mov nibbleFirstPart, bl ; change nibbleFirstPart to 0
		jmp set_ack
	nibble_second_part:	
		;nibble second part
		in 	al, dx
		lea bx, dataToReceive
		mov	si, DataOffset
		mov ah, [bx][si]
		and ah, 11110000b
		and al, 00001111b
		or  al, ah
		mov [bx][si], al
		mov bl, 1 
		mov nibbleFirstPart, bl ; change nibbleFirstPart to 1
		mov cx, DataOffset
		inc cx
		mov DataOffset, cx ; inc offset of data so we can read the next byte next time.
	set_ack:
		;first set busy to 0
		inc dx ; status ports
		in 	al, dx
		and al, 01111111b
		out dx, al ; set busy to 0
		;then set ack to 0
		in 	al, dx
		and al, 10111111b
		out dx, al ; set busy to 0
		jmp check_finish
	finish:
		mov al, 4c00h
		int 21h
		
end start 
