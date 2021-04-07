.386
.model flat
option casemap:none

.code
public _is_substring 
_is_substring proc
		push ebp
		push esi
		push edi
		push ebx
		push ecx
		mov ebp,esp
		mov ebx , [ebp+24]
		mov esi , ebx
		call _len
		mov edx,ecx ; edx = length of first string
		mov ebx , [ebp+28]
		mov edi , ebx
		call _len  ; ecx = length of second string
		xor ebx,ebx
	lbl4:	
		sub ecx,1
		sub edx,1
		mov esi , [ebp+24]
		mov edi , [ebp+28]
		add esi,ebx
		cmp edx,ecx
		jb lbl
		;length of first string is greater equal than second string
		push ecx
		cld ; clear direction flag
		repe cmpsb
		je lbl2
		pop ecx
		dec edx
		inc ebx
		jmp lbl4

	lbl: ; 
		mov al,0 ; set al to 0 return false
		jmp lbl3
	lbl2:
		pop ecx
		mov al,1 ; set al to 1 return true
	lbl3:
		pop ecx
		pop ebx
		pop edi
		pop esi
		pop ebp
		ret
_is_substring endp
_len proc ;ebx(input) = base address , ecx(output) = length of string
		push eax
		push ebx
		xor ecx,ecx
		mov al , 0
	lbl:	
		cmp [ebx] , al
		je lbl2
		inc ebx
		inc ecx
		jmp lbl
	lbl2:	
		pop ebx
		pop eax
		ret
_len endp
end


