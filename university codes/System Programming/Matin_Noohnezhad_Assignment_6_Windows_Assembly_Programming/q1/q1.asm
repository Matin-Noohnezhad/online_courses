.386
.model flat,stdcall
option casemap:none
WinMain proto :DWORD,:DWORD,:DWORD,:DWORD
include \masm32\include\windows.inc
include \masm32\include\user32.inc
include \masm32\include\kernel32.inc
include \masm32\include\gdi32.inc
includelib \masm32\lib\user32.lib
includelib \masm32\lib\kernel32.lib
includelib \masm32\lib\gdi32.lib

RGB macro red,green,blue
        xor eax,eax
        mov ah,blue
        shl eax,8
        mov ah,green
        mov al,red
endm

.data
ClassName db "SimpleWinClass",0
AppName  db "Question1 window",0
ButtonClassName db "button",0
ButtonText db "Transfer Text from 1 to 2",0
EditClassName1 db "edit",0
EditClassName2 db "edit",0
TestString db "Wow! I'm in an edit box now",0

.data?
hInstance HINSTANCE ?
CommandLine LPSTR ?
hwndButton HWND ?
hwndEdit1 HWND ?
hwndEdit2 HWND ?
buffer db 512 dup(?)

.const
ButtonID equ 1
EditID equ 2
IDM_HELLO equ 1
IDM_CLEAR equ 2
IDM_GETTEXT equ 3
IDM_EXIT equ 4
WINDOW_WIDTH equ 300
WINDOW_LENGTH equ 200

.code
start:
	invoke GetModuleHandle, NULL
	mov    hInstance,eax
	invoke GetCommandLine
	mov CommandLine,eax
	invoke WinMain, hInstance,NULL,CommandLine, SW_SHOWDEFAULT
	invoke ExitProcess,eax
WinMain proc hInst:HINSTANCE,hPrevInst:HINSTANCE,CmdLine:LPSTR,CmdShow:DWORD
	LOCAL wc:WNDCLASSEX
	LOCAL msg:MSG
	LOCAL hwnd:HWND
	mov   wc.cbSize,SIZEOF WNDCLASSEX
	mov   wc.style, CS_HREDRAW or CS_VREDRAW
	mov   wc.lpfnWndProc, OFFSET WndProc
	mov   wc.cbClsExtra,NULL
	mov   wc.cbWndExtra,NULL
	push  hInst
	pop   wc.hInstance
	RGB	  255,0,0;red color
	invoke    CreateSolidBrush, eax; use this method to create color for set hbrBackground 
	mov     wc.hbrBackground, eax
	mov   wc.lpszClassName,OFFSET ClassName
	invoke LoadIcon,NULL,IDI_APPLICATION
	mov   wc.hIcon,eax
	mov   wc.hIconSm,eax
	invoke LoadCursor,NULL,IDC_ARROW
	mov   wc.hCursor,eax
	invoke RegisterClassEx, addr wc
	invoke GetSystemMetrics, SM_CXSCREEN
	shr eax,1
	push ebx
	mov ebx,eax 
	invoke GetSystemMetrics, SM_CYSCREEN
	shr eax,1
	sub eax, WINDOW_WIDTH/2
	sub ebx, WINDOW_LENGTH/2
	INVOKE CreateWindowEx,WS_EX_CLIENTEDGE,ADDR ClassName,ADDR AppName,\
           WS_OVERLAPPEDWINDOW,ebx,\
           eax,WINDOW_WIDTH,WINDOW_LENGTH,NULL,NULL,\
           hInst,NULL
	pop ebx
	mov   hwnd,eax
	INVOKE ShowWindow, hwnd,SW_SHOWNORMAL
	INVOKE UpdateWindow, hwnd
	.WHILE TRUE
                INVOKE GetMessage, ADDR msg,NULL,0,0
                .BREAK .IF (!eax)
                INVOKE TranslateMessage, ADDR msg
                INVOKE DispatchMessage, ADDR msg
	.ENDW
	mov     eax,msg.wParam
	ret
WinMain endp
WndProc proc hWnd:HWND, uMsg:UINT, wParam:WPARAM, lParam:LPARAM
	.IF uMsg==WM_DESTROY
		invoke PostQuitMessage,NULL
	.ELSEIF uMsg==WM_CREATE
		invoke CreateWindowEx,WS_EX_CLIENTEDGE, ADDR EditClassName1,NULL,\
                        WS_CHILD or WS_VISIBLE or WS_BORDER or ES_LEFT or\
                        ES_AUTOHSCROLL,\
                        25,35,100,25,hWnd,EditID,hInstance,NULL
		mov  hwndEdit1,eax
		invoke SetFocus, hwndEdit1
		invoke CreateWindowEx,WS_EX_CLIENTEDGE, ADDR EditClassName2,NULL,\
                        WS_CHILD or WS_VISIBLE or WS_BORDER or ES_LEFT or\
                        ES_AUTOHSCROLL,\
                        150,35,100,25,hWnd,EditID,hInstance,NULL
		mov  hwndEdit2,eax
		invoke SetFocus, hwndEdit2
		invoke CreateWindowEx,NULL, ADDR ButtonClassName,ADDR ButtonText,\
                        WS_CHILD or WS_VISIBLE or BS_DEFPUSHBUTTON,\
                        55,90,180,25,hWnd,ButtonID,hInstance,NULL
		mov  hwndButton,eax
	.ELSEIF uMsg==WM_COMMAND
		mov eax,wParam
		.IF ax==ButtonID
			shr eax,16
			.IF ax==BN_CLICKED
				invoke GetWindowText,hwndEdit1,ADDR buffer,512
				invoke SetWindowText,hwndEdit2,ADDR buffer
			.ENDIF
		.ENDIF
	.ELSE
		invoke DefWindowProc,hWnd,uMsg,wParam,lParam
		ret
	.ENDIF
	xor    eax,eax
	ret
WndProc endp
end start
