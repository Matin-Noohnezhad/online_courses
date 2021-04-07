.386
.model flat,stdcall
option casemap:none
include \masm32\include\windows.inc
include \masm32\include\user32.inc
include \masm32\include\kernel32.inc
include \masm32\include\gdi32.inc

includelib \masm32\lib\user32.lib
includelib \masm32\lib\kernel32.lib
includelib \masm32\lib\gdi32.lib

WinMain     proto :DWORD,:DWORD,:DWORD,:DWORD
CALC_FUNC   proto :QWORD
CALC_PIMULTIPLIER  proto :QWORD
CALC_SIGN  proto :DWORD
XDIV100     proto :DWORD


RGB macro red,green,blue
        xor eax,eax
        mov ah,blue
        shl eax,8
        mov ah,green
        mov al,red
endm

.const
    startY  dd 180
    mulY    dd -30   
    one     dd 1
    five    dd 5
    hundred dd 100
    pi      dq 3.14   

.data
    ClassName       db "SimpleWinClass",0
    AppName         db "LOG5a + 5*SINa",0
    aInput          dq 1.0
    signNumber      dd 1

.data?
    hInstance       dd ?
    CommandLine     dd ?
    x               dq ?
    y               dq ?
    r               dq ?
    sina            dq ?
    loga5           dq ?
    piMultiplier    dd ?

    ans             dd ?


.code
start:
	invoke GetModuleHandle, NULL
	mov    hInstance,eax
	invoke GetCommandLine
	mov    CommandLine,eax
	invoke WinMain, hInstance, NULL, CommandLine, SW_SHOWDEFAULT
	invoke ExitProcess, eax

WinMain proc hInst:HINSTANCE, hPrevInst:HINSTANCE, CmdLine:LPSTR, CmdShow:DWORD
	LOCAL wc:WNDCLASSEX
	LOCAL msg:MSG
	LOCAL hwnd:HWND
	
	mov   wc.cbSize, SIZEOF WNDCLASSEX
	mov   wc.style, CS_HREDRAW or CS_VREDRAW
	mov   wc.lpfnWndProc, OFFSET WndProc
	mov   wc.cbClsExtra, NULL
	mov   wc.cbWndExtra, NULL
	push  hInstance
	pop   wc.hInstance
	mov   wc.hbrBackground, COLOR_WINDOW+1
	mov   wc.lpszMenuName, NULL
	mov   wc.lpszClassName, OFFSET ClassName
	invoke LoadIcon, NULL, IDI_APPLICATION
	mov   wc.hIcon, eax
	mov   wc.hIconSm, eax
	invoke LoadCursor, NULL, IDC_ARROW
	mov   wc.hCursor, eax
	invoke RegisterClassEx, addr wc
	INVOKE CreateWindowEx,NULL,ADDR ClassName,ADDR AppName,\
           WS_OVERLAPPEDWINDOW,0,\
           0,CW_USEDEFAULT,CW_USEDEFAULT,NULL,NULL,\
           hInst,NULL
	mov   hwnd, eax

	invoke ShowWindow, hwnd, SW_SHOWNORMAL
	invoke UpdateWindow, hwnd
	.WHILE TRUE
		invoke GetMessage, ADDR msg, NULL, 0, 0
		.BREAK .IF (!eax)
		invoke TranslateMessage, ADDR msg
		invoke DispatchMessage, ADDR msg
	.ENDW
	mov     eax,msg.wParam
	ret
WinMain endp

WndProc proc hWnd:HWND, uMsg:UINT, wParam:WPARAM, lParam:LPARAM
        LOCAL ps:     PAINTSTRUCT
        LOCAL hdc:    HDC

	.if uMsg==WM_DESTROY
		invoke PostQuitMessage, NULL
    .ELSEIF uMsg==WM_SIZE
         INVOKE BeginPaint, hWnd, ADDR ps
		 mov    hdc, eax

        mov eax, 1; eax <-- x
  plot_lbl: 
        invoke XDIV100, eax
        invoke CALC_PIMULTIPLIER, aInput
        invoke CALC_SIGN, piMultiplier
        invoke CALC_FUNC, aInput
        mov edx, ans
        push eax
        invoke SetPixel, hdc, eax, edx, 0ff00ffh        
        pop eax
        add eax, 1
        cmp eax, 800
        jnz plot_lbl
	.else
		invoke DefWindowProc, hWnd, uMsg, wParam, lParam		
		ret
	.endif
	xor eax,eax
	ret
WndProc endp
XDIV100 proc input:DWORD
    finit
    fild input
    fild hundred
    fdiv
    fstp aInput
    fwait
    ret
XDIV100 endp
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
    mov signNumber,eax 
    jmp sgn_lbl2
sgn_lbl:
    mov eax, -1
    mov signNumber, eax
sgn_lbl2:
    pop eax
    ret
CALC_SIGN  endp
CALC_FUNC proc input:QWORD
    ;calc loga5
    finit
    fild one
    fild five
    fyl2x
    fild one
    fdiv st(0),st(1)
    fld input
    fyl2x
    fstp loga5
    ;calc r
    fld input
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
    fild signNumber
    fmul
    fstp sina
    ;loga5 + 5*sina
    fild five
    fld sina
    fmul
    fld loga5
    fadd ; st(0) = loga5 + 5*sina
    fild mulY
    fmul st(0),st(1); st(0) = ans* mulY
    fild startY
    fadd; st(0) = (ans* mulY) + startY
    fistp ans
    fwait
    ret
CALC_FUNC endp
end start