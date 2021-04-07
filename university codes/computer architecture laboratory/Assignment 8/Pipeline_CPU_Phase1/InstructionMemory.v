`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:55:43 05/25/2018 
// Design Name: 
// Module Name:    InstructionMemory 
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
module InstructionMemory(address, instruction, startin);
output [31:0]instruction;
input [31:0]address;
input startin;
reg [7:0] instructions [83:0];

always @(posedge startin)
begin
{instructions[0],instructions[1],instructions[2],instructions[3]} <= 32'b10001110000010000000000000000000;
{instructions[4],instructions[5],instructions[6],instructions[7]} <= 32'b00000001000000001000100000100000;
{instructions[8],instructions[9],instructions[10],instructions[11]} <= 32'b10001110000010000000000000000100;
{instructions[12],instructions[13],instructions[14],instructions[15]} <= 32'b00000001000100011000100000100000;
{instructions[16],instructions[17],instructions[18],instructions[19]} <= 32'b10001110000010000000000000001000;
{instructions[20],instructions[21],instructions[22],instructions[23]} <= 32'b00000001000100011000100000100000;
{instructions[24],instructions[25],instructions[26],instructions[27]} <= 32'b10001110000010000000000000001100;
{instructions[28],instructions[29],instructions[30],instructions[31]} <= 32'b00000001000100011000100000100000;
{instructions[32],instructions[33],instructions[34],instructions[35]} <= 32'b10001110000010000000000000010000;
{instructions[36],instructions[37],instructions[38],instructions[39]} <= 32'b00000001000100011000100000100000;
{instructions[40],instructions[41],instructions[42],instructions[43]} <= 32'b10001110000010000000000000010100;
{instructions[44],instructions[45],instructions[46],instructions[47]} <= 32'b00000001000100011000100000100000;
{instructions[48],instructions[49],instructions[50],instructions[51]} <= 32'b10001110000010000000000000011000;
{instructions[52],instructions[53],instructions[54],instructions[55]} <= 32'b00000001000100011000100000100000;
{instructions[56],instructions[57],instructions[58],instructions[59]} <= 32'b10001110000010000000000000011100;
{instructions[60],instructions[61],instructions[62],instructions[63]} <= 32'b00000001000100011000100000100000;
{instructions[64],instructions[65],instructions[66],instructions[67]} <= 32'b10001110000010000000000000100000;
{instructions[68],instructions[69],instructions[70],instructions[71]} <= 32'b00000001000100011000100000100000;
{instructions[72],instructions[73],instructions[74],instructions[75]} <= 32'b10001110000010000000000000100100;
{instructions[76],instructions[77],instructions[78],instructions[79]} <= 32'b00000001000100011000100000100000;
{instructions[80],instructions[81],instructions[82],instructions[83]} <= 32'b10101110000100010000000000101000;
end
assign instruction = {instructions[address],instructions[address+1],instructions[address+2],instructions[address+3]};


endmodule
