`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    00:10:00 04/27/2018 
// Design Name: 
// Module Name:    Question3 
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
module Question3(A,B,C,D,out);

input A,B,C,D;
output out;

assign out = ((A&B&C) | (A&B&D) | (A&C&D) | (B&C&D))?1:0;

endmodule
