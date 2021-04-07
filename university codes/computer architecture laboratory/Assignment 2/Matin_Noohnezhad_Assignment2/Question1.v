`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:21:40 04/26/2018 
// Design Name: 
// Module Name:    Question1 
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
module Question1(A,B,sum,cout);

input A,B;
output sum,cout;

assign{cout,sum} = A+B;

endmodule
