`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:01:47 07/09/2018 
// Design Name: 
// Module Name:    Mux3to1 
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
module Mux3to1(in1,in2,in3,selector,out);
input [31:0]in1;
input [31:0]in2;
input [31:0]in3;
input [1:0]selector = 2'b0;
output [31:0]out;


assign out = (selector == 2'b0)? in1:(selector == 2'b01)? in2:in3;



endmodule
