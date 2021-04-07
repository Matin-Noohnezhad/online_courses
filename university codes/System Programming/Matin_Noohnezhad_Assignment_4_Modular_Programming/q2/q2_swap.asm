.386
.model flat

option casemap:none

.code
public _myswap
_myswap proc
push ebp  
push ebx
push esi
mov ebp,esp 
mov ebx , [ebp + 16]
mov edx , [ebx]
push edx
mov ebp,esp 
mov esi , [ebp + 24]
mov edx , [esi]
mov [ebx] , edx
pop edx
mov [esi] , edx
pop esi
pop ebx
pop ebp
ret 
_myswap endp
end


