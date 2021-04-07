`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:58:58 07/01/2018 
// Design Name: 
// Module Name:    ALUControl 
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
module ALUControl(Func, Aluop, Alucontrol);
input [5:0]Func;
input [1:0]Aluop;
output [3:0]Alucontrol;

//for our cpu alucontrol is always 0010(because lw , sw and add alufunction are all add)
assign Alucontrol = 4'b0010;



endmodule
