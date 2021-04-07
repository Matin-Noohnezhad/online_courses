`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    02:47:02 07/10/2018 
// Design Name: 
// Module Name:    EXMEM 
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
module EXMEM(aluresult, zero, jump, branch, memread, memtoreg, memwrite, regwrite, addOut2, readData2,muxOut1,clk, aluresultOut, zeroOut, jumpOut, branchOut, memreadOut, memtoregOut, memwriteOut, regwriteOut, addOut2Out, readData2Out,muxOut1Out);

input [31:0]aluresult;
input zero;
input jump;
input branch;
input memread;
input memtoreg;
input memwrite;
input regwrite;
input [31:0]addOut2;
input [31:0]readData2;
input [4:0]muxOut1;
input clk;

output reg [31:0]aluresultOut;
output reg zeroOut;
output reg [31:0]addOut2Out;
output reg [31:0]readData2Out;
output reg [4:0]muxOut1Out;
output reg jumpOut;
output reg branchOut;
output reg memreadOut;
output reg memtoregOut;
output reg memwriteOut;
output reg regwriteOut;

always @(posedge clk)
begin
aluresultOut <= aluresult;
zeroOut <=zero;
jumpOut<=jump;
branchOut<=branch;
memreadOut<=memread;
memtoregOut<=memtoreg;
memwriteOut<=memwrite;
regwriteOut<=regwrite;
addOut2Out<=addOut2;
readData2Out<=readData2;
muxOut1Out<=muxOut1;
end

endmodule
