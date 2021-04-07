.386
.model flat,stdcall
option casemap:none

include \masm32\include\windows.inc
include \masm32\include\user32.inc
include \masm32\include\kernel32.inc
include \masm32\include\comdlg32.inc
include \masm32\include\msvcrt.inc
includelib \masm32\lib\user32.lib
includelib \masm32\lib\kernel32.lib
includelib \masm32\lib\comdlg32.lib
includelib \masm32\lib\msvcrt.lib

.const
    memSize equ 5
.data 
    testPrint   db "The word is --> %s",10,0
    testPrint2   db "The number of read bytes is --> %d",10,0
    filename    db "matin.exe",0
    prt1        db "First segment offset is: %d", 10, 0   
    prt2        db "First relative address is: %d", 10, 0

.data?
    pMemory     dd      ? 
    hFile       dd      ?
    SizeReadWrite DWORD ?
    pPEHeader   dd      ?
    pSizeOfHeaders  dd  ?
    sizeOfHeaders   dd  ?
    pAddressOfEntryPoint    dd  ?
    addressOfEntryPoint    dd  ?


.code
start:
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ;GlobalAlloc (allocate block of memory)
    invoke GlobalAlloc,GMEM_FIXED or GMEM_ZEROINIT ,memSize ; it returns a pointer to memmory that allocated
    mov pMemory, eax
    ;xxxxxxxxxxx GlobalLock xxxxxxxxxxxxx
    ;CreateFile
    invoke CreateFile, addr filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_ARCHIVE, NULL
    mov hFile, eax
    ;SetFilePointer
    invoke SetFilePointer, hFile, 3ch , 0, FILE_BEGIN ; set the pointer to the start of the file 
    ;ReadFile
    invoke ReadFile, hFile, pMemory, memSize-1, addr SizeReadWrite, NULL
    mov ebx, pMemory
    mov eax, [ebx] ; pointer to PEHeader
    mov pPEHeader, eax
    add eax, 84    ; pointer to sizeOfHeaders
    mov pSizeOfHeaders, eax
    invoke SetFilePointer, hFile, pSizeOfHeaders , 0, FILE_BEGIN ; set the pointer to the start of the file 
    invoke ReadFile, hFile, pMemory, memSize-1, addr SizeReadWrite, NULL
    mov ebx, pMemory
    mov eax, [ebx] ; size of headers
    mov sizeOfHeaders, eax
    ;;;;
    mov eax, pSizeOfHeaders
    sub eax, 44 ; pointer to address of entry point
    mov pAddressOfEntryPoint, eax
    invoke SetFilePointer, hFile, pAddressOfEntryPoint , 0, FILE_BEGIN ; set the pointer to the start of the file 
    invoke ReadFile, hFile, pMemory, memSize-1, addr SizeReadWrite, NULL
    mov ebx, pMemory
    mov eax, [ebx] ; size of headers
    mov addressOfEntryPoint, eax
    ;xxxxxxxxxxx GlobalUnlock xxxxxxxxxxx

    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ; invoke crt_printf, offset testPrint, pMemory
    ; invoke crt_printf, offset testPrint2, SizeReadWrite
    invoke crt_printf, offset prt1, sizeOfHeaders
    invoke crt_printf, offset prt2, addressOfEntryPoint
    ;CloseHandle
    invoke CloseHandle, hFile
    ;GlobalFree
    invoke GlobalFree, pMemory
    invoke ExitProcess,0

end start


