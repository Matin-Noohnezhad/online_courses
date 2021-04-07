`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:42:55 04/26/2018 
// Design Name: 
// Module Name:    Question2B 
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
module Question2B(A,B,cin,S,cout);

input A,B,cin;
output S,cout;
wire [2:0]n;
wire zero,one,two,three,four,five,six,seven;

assign n[0] = A;
assign n[1] = B;
assign n[2] = cin;

Question2A decoder(n,zero,one,two,three,four,five,six,seven);

assign S = ((one == 1)|(two == 1)|(four == 1 )|(seven == 1))?1:0;
assign cout = ((three == 1)|(five == 1 )|(six == 1)|(seven == 1))?1:0;

endmodule
