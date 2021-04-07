.386
.model flat

option casemap:none 

.code

public _func

_func proc
push ebp
mov ebp,esp
push ebx
push ecx
mov eax, [ebp+8]
mov ecx , eax;ecx=a
shl eax,3; eax = 8*a
sub eax,ecx ; eax = 7*a
mov ebx, [ebp+12]
mov ecx,ebx;ecx=b
shl ebx,2; ebx = 4*b
sub ebx,ecx ; ebx = 3*b
add eax,ebx
pop ecx
pop ebx
pop ebp
ret
_func endp

end