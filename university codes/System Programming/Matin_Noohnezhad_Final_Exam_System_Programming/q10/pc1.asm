.model small

.stack 64

.data
	dataToTransfer 		db  1000 dup(5) ; data to send
	lpt1PinNo 			dw  03BCh ; 03BC --> data, 03BD --> status, 03BE --> control 
	nibbleFirstPart 	db	1 ; Is it first part of nibble or second part of it.
	DataOffset			dw	0 ; Which offset of data are we sending right now.
	
.code
	start: 
		mov		dx, @data
		mov		ds, dx
		mov 	dx, lpt1PinNo
		inc 	dx
		inc 	dx  ;control port
		in		al, dx
		and		al, 11011111b 
		out		dx, al	; make it forward direction by set bit 5 of control port to zero
		dec 	dx	;status port
	check_finish:
		cmp DataOffset, 1000
		jz  finish
	busy_check:
		in  	al, dx
		test	al, 10000000b
		jnz		busy_check	; if busy check again until the busy bit is set to zero
		;no busy
		dec		dx	;data port
		lea 	bx, dataToTransfer
		mov		si, DataOffset
		mov		al, [bx][si]
		mov 	bl, nibbleFirstPart
		test	bl, 1
		jz		nibble_second_part
		;nibble first part
		or 	al, 11100000b ; most significant 3 bit is useless.(No of communcation bits are 5)
		out dx, al
		xor bl, bl
		mov nibbleFirstPart, bl ; change nibbleFirstPart to 0
		jmp set_strobe
	nibble_second_part:	
		;nibble second part
		mov cx, 4
		shr al, cx; we want to transfer the high nibble
		or 	al, 11100000b ; most significant 3 bit is useless.(No of communcation bits are 5)
		out dx, al
		mov bl, 1 
		mov nibbleFirstPart, bl ; change nibbleFirstPart to 1
		mov cx, DataOffset
		inc cx
		mov DataOffset, cx ; inc offset of data so we can read the next byte next time.
	set_strobe:
		inc dx
		inc dx ; control ports
		in 	al, dx
		and al, 11111110b
		out dx, al ; set strobe to zero
		;check Ack bit
		dec dx ; status ports
	check_ack:
		in al, dx
		test al, 01000000b
		jnz check_ack
		or al, 01000000b ;
		out dx, al; set ack to 1 until pc2 turn it back to 0
		jmp check_finish
	finish:
		mov al, 4c00h
		int 21h
		
end start 
