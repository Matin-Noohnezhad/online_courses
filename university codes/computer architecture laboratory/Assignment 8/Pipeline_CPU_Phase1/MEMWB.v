`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    02:47:22 07/10/2018 
// Design Name: 
// Module Name:    MEMWB 
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
module MEMWB(jump, branch, memtoreg, regwrite, readDataMemory, aluResultEXMEM, muxOutEXMEM,zeroEXMEM,clk,jumpOut, branchOut, memtoregOut, regwriteOut, readDataMemoryOut, aluResultEXMEMOut, muxOutEXMEMOut,zeroEXMEMOut);
input jump;
input branch;
input memtoreg;
input regwrite;
input [31:0]readDataMemory;
input [31:0]aluResultEXMEM;
input [4:0]muxOutEXMEM;
input zeroEXMEM;
input clk;
output reg jumpOut;
output reg branchOut;
output reg memtoregOut;
output reg regwriteOut;
output reg [31:0]readDataMemoryOut;
output reg [31:0]aluResultEXMEMOut;
output reg [4:0]muxOutEXMEMOut;
output reg zeroEXMEMOut;


always @(posedge clk)
begin
jumpOut <=jump;
branchOut <=branch;
memtoregOut <=memtoreg;
regwriteOut <=regwrite;
readDataMemoryOut <=readDataMemory;
aluResultEXMEMOut <=aluResultEXMEM;
muxOutEXMEMOut <=muxOutEXMEM;
zeroEXMEMOut <=zeroEXMEM;
end

endmodule
