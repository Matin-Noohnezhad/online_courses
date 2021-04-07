.386
.model flat,c

option casemap:none

.code
public myadd 
myadd proc  a:dword, b:dword
xor eax,eax
add eax , a
add eax , b
ret 
myadd endp
end
