`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:24:14 07/09/2018 
// Design Name: 
// Module Name:    HDU 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module HDU(IFIDrs,IFIDrt,IDEXmemRead,IDEXrt,pcWrite,IFIDwrite,IDEXzero);
//if (ID/EX.MemRead and
//((ID/EX.RegisterRt = IF/ID.RegisterRs) or
//(ID/EX.RegisterRt = IF/ID.RegisterRt)))
//stall the pipeline
input [4:0]IFIDrs;
input [4:0]IFIDrt;
input IDEXmemRead;
input [4:0]IDEXrt;
output pcWrite;
output IFIDwrite;
output IDEXzero;

assign pcWrite = (IDEXmemRead && ((IDEXrt == IFIDrs) || (IDEXrt == IFIDrt)))?1:0;
assign IFIDwrite = (IDEXmemRead && ((IDEXrt == IFIDrs) || (IDEXrt == IFIDrt)))?1:0;
assign IDEXzero = (IDEXmemRead && ((IDEXrt == IFIDrs) || (IDEXrt == IFIDrt)))?1:0;

endmodule
