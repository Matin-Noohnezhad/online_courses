`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:35:08 07/01/2018 
// Design Name: 
// Module Name:    Mux_5 
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
module Mux_5(input1,input2,op,out);
input [4:0]input1;
input [4:0]input2;
input op;
output [4:0]out;

assign out = (op )? input2 : input1;

endmodule
