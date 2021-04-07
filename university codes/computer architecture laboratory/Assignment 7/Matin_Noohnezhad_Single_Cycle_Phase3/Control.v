`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:17:12 07/01/2018 
// Design Name: 
// Module Name:    Control 
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
module Control(instruction31_26, regdst, jump, branch, memread, memtoreg, aluop, memwrite, alusrc, regwrite);
input [5:0] instruction31_26;
output regdst;
output jump;
output branch;
output memread;
output memtoreg;
output memwrite;
output alusrc;
output regwrite;
output [1:0]aluop;


assign regdst = (instruction31_26 == 6'b000000) ? 1:0;
assign jump = 0;
assign branch = 0;
assign memread = (instruction31_26 == 6'b100011) ? 1:0;
assign memtoreg = (instruction31_26 == 6'b100011) ? 1:0;
assign memwrite = (instruction31_26 == 6'b101011) ? 1:0;
assign alusrc = (instruction31_26 == 6'b000000) ? 0:1;
assign regwrite = (instruction31_26 == 6'b101011) ? 0:1;
assign aluop = (instruction31_26 == 6'b000000) ? 2'b10:2'b00;

endmodule
