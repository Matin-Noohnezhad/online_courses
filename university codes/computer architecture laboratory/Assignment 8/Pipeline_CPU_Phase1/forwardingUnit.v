`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    00:09:08 07/10/2018 
// Design Name: 
// Module Name:    ForwardingUnit 
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
module ForwardingUnit(EXMEMregWrite,EXMEMrd,IDEXrt,IDEXrs,MEMWBregWrite,MEMWBrt,forward_1,forward_2);

input EXMEMregWrite;
input [4:0]EXMEMrd;
input MEMWBregWrite;
input [4:0]MEMWBrt;
input [4:0]IDEXrt;
input [4:0]IDEXrs;
output [1:0]forward_1;
output [1:0]forward_2;

assign forward_1 = (MEMWBregWrite && (MEMWBrt == IDEXrs) && (MEMWBrt != 0))? 2'b10:(EXMEMregWrite && (EXMEMrd == IDEXrs) && (MEMWBrt != 0))? 2'b01:2'b0;
assign forward_2 = (MEMWBregWrite && (MEMWBrt == IDEXrt) && (MEMWBrt != 0))? 2'b10:(EXMEMregWrite && (EXMEMrd == IDEXrt) && (MEMWBrt != 0))? 2'b01:2'b0;



endmodule
