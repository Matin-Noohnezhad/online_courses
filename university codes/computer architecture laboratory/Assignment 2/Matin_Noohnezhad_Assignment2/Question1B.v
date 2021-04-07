`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:47:35 04/26/2018 
// Design Name: 
// Module Name:    Question1B 
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
module Question1B(A,B,answer);

input [1:0]A;
input [1:0]B;
output [3:0]answer;
wire cout1,cout2;

assign answer[0] = A[0] & B[0];
Question1 HA1((A[0] & B[1]) , (A[1] & B[0]) , answer[1] , cout1);
Question1 HA2((A[1] & B[1]) , cout1 , answer[2] , answer[3]);

endmodule
