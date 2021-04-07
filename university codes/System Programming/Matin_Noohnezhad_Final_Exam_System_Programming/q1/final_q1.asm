.386
.model flat,stdcall
option casemap:none
include \masm32\include\windows.inc
include \masm32\include\user32.inc
include \masm32\include\kernel32.inc
include c:\masm32\include\msvcrt.inc

includelib \masm32\lib\user32.lib
includelib \masm32\lib\kernel32.lib
includelib c:\masm32\lib\msvcrt.lib

CALC_FUNC           proto :QWORD, :QWORD
CALC_PIMULTIPLIER   proto :QWORD
CALC_SIGN           proto :DWORD
CALC_POWER          proto :DWORD, :DWORD

.const
    one     dd 1
    three   dd 3
    six     dd 6
    pi      dq 3.14   

.data
    prntAns            db "The answer is = %lf",10,0
	prntSin            db "The sin is = %lf",10,0
	prntCos            db "The cos is = %lf",10,0
	prntXPowY          db "The x power y is = %lf",10,0
	prntDoublePow	   db "The double power is = %lf",10,0
	prntPow	           db "The real power is = %d",10,0
	prntPowerResult    db "The power result is = %d",10,0
    aInput          dq 1.0
    signNumberSin      dd 1
    signNumberCos      dd -1
    xInput              dq 1.57; x
    yInput              dq 2.0   ; y


.data?
    x               dq ?
    y               dq ?
    r               dq ?
    sina            dq ?
    cosa            dq ?
    XPowY           dq ?
    testPow           dq ?
    piMultiplier    dd ?
    doublePOWER       dq ?
    realPOWER       dd ?
    RealPowerResult dd ?

    ans             dq ?


.code
start:
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        invoke CALC_PIMULTIPLIER, xInput ;set piMultiplier
        invoke CALC_SIGN, piMultiplier   ;set signNumber
        invoke CALC_FUNC, xInput, yInput
        invoke crt_printf, offset prntAns, ans
		invoke crt_printf, offset prntCos, cosa
		invoke crt_printf, offset prntSin, sina
		invoke crt_printf, offset prntXPowY, XPowY
		; invoke crt_printf, offset prntXPowY, testPow
		; invoke crt_printf, offset prntDoublePow, doublePOWER
		; invoke crt_printf, offset prntPow, realPOWER
		; invoke crt_printf, offset prntPowerResult, RealPowerResult
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	invoke ExitProcess, 0

CALC_PIMULTIPLIER  proc input:QWORD
    finit
    fld input
    fld pi
    fdiv
    fistp piMultiplier
    fwait
    ret
CALC_PIMULTIPLIER  endp
CALC_SIGN  proc input:DWORD
        push eax
        mov eax, input
        and eax, 1
        cmp eax, 1
        jz sgn_lbl ; input is even number
        mov eax, 1 ; input is odd number
        mov ecx, -1
        mov signNumberSin,eax 
        mov signNumberCos,ecx 
        jmp sgn_lbl2
    sgn_lbl:
        mov eax, -1
        mov ecx, 1
        mov signNumberSin, eax
        mov signNumberCos, ecx
    sgn_lbl2:
        pop eax
        ret
CALC_SIGN  endp
CALC_FUNC proc xinput:QWORD, yinput:QWORD
    finit
    fld yinput
    fld xinput
    fyl2x; now st(0) = y * log2x
    fist realPOWER
    fst doublePOWER
    mov eax,2 
    invoke CALC_POWER, eax, realPOWER; 2 ^ POWER ==> RealPowerResult
    fld doublePOWER
    fild realPOWER
    fsubp st(1),st(0)
	F2XM1; 2 ^ (something between -1 and 1) ==> st(0)
	fild one
    faddp; now st(0) = 2^(y * log2x)
    fst testPow
    fild RealPowerResult
    fmulp
    fstp XPowY
    ;calc r
    fld xinput
    fptan
    fstp x
    fstp y
    ;;;;;;;;;;;
    fld x
    fmul st(0),st(0)
    fld y
    fmul st(0),st(0)
    fadd st(0),st(1)
    fsqrt
    fstp r
    ;calc sin
    fld r
    fld y
    fdiv st(0),st(1)
    fild signNumberSin
    fmul
    fstp sina
    ;calc cos
    fld r
    fld x
    fdiv st(0),st(1)
    fild signNumberCos
    fmul
    fstp cosa
    ;6(x^y) + 3COSx * SINx * x
    fild three
    fld sina
    fmulp
    fld cosa
    fmulp
    fld xinput
    fmulp; now st(0) = 3COSx * SINx * x
    fild six
    fld XPowY
    fmulp; now st(0) = 6 * (x^y)
    faddp; now st(0) = 6 * (x^y) + 3COSx * SINx * x
    fstp ans
    fwait
    ret
CALC_FUNC endp
CALC_POWER PROC xin:DWORD, yin:DWORD
    local XINPUT: DWORD
	mov eax, xin
    mov XINPUT, eax
    mov eax, 1
    mov RealPowerResult, eax 
    mov ecx, yin
    cmp ecx, 0
    jz fin_lbl
    mov eax, 2
    mov RealPowerResult, eax 
power_lbl:
    cmp ecx, 1
    jz fin_lbl
    fild XINPUT
    fild RealPowerResult
    fmulp st(1),st(0)
    fistp RealPowerResult
    dec ecx
    jmp power_lbl
fin_lbl:
    ret
CALC_POWER endp
end start