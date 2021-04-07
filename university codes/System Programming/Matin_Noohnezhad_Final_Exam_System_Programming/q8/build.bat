del %1.obj %1.exe
ml  /c  /coff  /Cp %1.asm
link /SUBSYSTEM:CONSOLE  /LIBPATH:c:\masm32\lib  %1.obj
.\%1.exe