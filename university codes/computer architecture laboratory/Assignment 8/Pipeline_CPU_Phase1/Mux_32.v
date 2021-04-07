`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:57:32 06/01/2018 
// Design Name: 
// Module Name:    Mux_32 
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
module Mux_32(input1, input2, op, out);

input [31:0]input1;
input [31:0]input2;
input op;
output [31:0]out;

assign out = (op )? input2 : input1;


endmodule
