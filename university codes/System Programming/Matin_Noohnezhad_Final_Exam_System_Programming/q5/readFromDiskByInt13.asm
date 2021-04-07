; This code is for Hard Disk with FAT32 format, and it reads the second cluster of
; first partition of that disk.
; you should assemble this with /tiny to get a COM file.
;;;;;;;;
; first find out reserved bytes after MBR in first partition and store that in
; hidden sectors variable in memory, then we should go to that Boot sector of 
; first partition and find out the size of reserved sectors. then read the size of 
; FAT in sectors and multiply it by two cause it have two FAT then add all of the together
; to find out the address of root directory which is the first cluster of first partition
; then add it by 16 to get the address of second cluster in first partition and read that
; and store it in memory and after that you could save that in a text file.

.model small

.stack 64

.data
		head_per_cylinder	dw		16 ; I don't know the exact number.
		sector_per_head 	dw		256 ; I don't know the exact number.
.data?
		target_cluster 	db 		4096 dup(?)
		sectorNo		dd 				?
		byteNO			dd 				?
		dwordOutput		dd				?
		hiddenSectors	dd				?
		reservedSectors	dd				?
		
.code 
start:
	assume ds:@data
	mov bx, @data
	mov ds,bx
	;
	mov [byteNo], 1C6h ; read first sector of first partition which is boot sector of it
	call readDWordFromDisk
	mov ax, [dwordOutput]
	mov [hiddenSectors], ax
	mov ax, [dwordOutput+2]
	mov [hiddenSectors+2], ax
	;...	
	;
	call LBAtoCHSConversion
	mov dl, 80h ; My first hard disk. (it should has FAT32 format)
	mov al, 8	; Read 8 sector which means 1 cluster.
	mov bx, offset target_cluster ; 
	mov ah, 2	; int 13h service --> read sectors from drive
	int 13h
	mov ax, 4c00h
	int 21h		; invoke ExitProcess, 0

LBAtoCHSConversion proc ;input in sectorNo in memory ------- output in ch <-- cylinder, cl <-- sector, dh <-- head
		;TODO
		;mov ch, 0
		;mov dh, 0
		;mov cl, 1
		ret
LBAtoCHSConversion endp
readDWordFromDisk proc ;input in byteNo in memory --------- dword 
		;first you must convert the byte to sector and read that sector by
		;int 13h then read that specific byte from that sector then you can
		;store that dword in dwordOutput
		ret
readDWordFromDisk endp

end start








