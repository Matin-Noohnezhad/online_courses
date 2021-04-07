`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:07:50 04/26/2018 
// Design Name: 
// Module Name:    Question2A 
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
module Question2A(A,zero,one,two,three,four,five,six,seven);

input [2:0]A;
output zero,one,two,three,four,five,six,seven;

assign zero = (~A[2])&(~A[1])&(~A[0]);
assign one = (~A[2])&(~A[1])&A[0];
assign two = (~A[2])&A[1]&(~A[0]);
assign three = (~A[2])&A[1]&A[0];
assign four = A[2]&(~A[1])&(~A[0]);
assign five = A[2]&(~A[1])&A[0];
assign six = A[2]&A[1]&(~A[0]);
assign seven = A[2]&A[1]&A[0];

endmodule
