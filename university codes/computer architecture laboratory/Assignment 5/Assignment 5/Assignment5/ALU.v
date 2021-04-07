`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:55:19 05/25/2018 
// Design Name: 
// Module Name:    ALU 
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
module ALU(in1, in2, operation, out, zero);
input [31:0]in1;
input [31:0]in2;
input [3:0]operation;
output [31:0]out;
output zero;

//and =0 , or = 1, add =2, sub=5,slt = 6,nor=12
//assign zero = 0;
//if(operation == 4'b10)
//assign out = in1 +in2;
//else if(operation == 4'b101)
//begin
//assign out = in1-in2;
//if(out == 32'b0)
//assign zero = 1;
//end
assign out =(operation == 4'b10) ? in1+in2:(operation == 4'b101) ? in1-in2: 32'b10101010010101;
assign zero =(out == 32'b0) ? 1:0;


endmodule
