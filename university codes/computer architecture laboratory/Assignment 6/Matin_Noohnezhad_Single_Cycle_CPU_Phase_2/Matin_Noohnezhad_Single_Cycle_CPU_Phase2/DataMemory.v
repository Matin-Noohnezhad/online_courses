`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:09:19 06/01/2018 
// Design Name: 
// Module Name:    DataMemory 
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
module DataMemory(Address, WriteData, MemWrite, MemRead, ReadData, startin, clk);

input [31:0]Address;
input [31:0]WriteData;
input MemWrite;
input MemRead;
input clk;
input startin;
output [31:0]ReadData;
reg [7:0]MemoryDatas [43:0];

always @(posedge clk)
	begin
	if(startin)
	begin
		//first 10 memory
		{MemoryDatas[0],MemoryDatas[1],MemoryDatas[2],MemoryDatas[3]} <= 32'b1;
		{MemoryDatas[4],MemoryDatas[5],MemoryDatas[6],MemoryDatas[7]} <= 32'b10;
		{MemoryDatas[8],MemoryDatas[9],MemoryDatas[10],MemoryDatas[11]} <= 32'b11;
		{MemoryDatas[12],MemoryDatas[13],MemoryDatas[14],MemoryDatas[15]} <= 32'b100;
		{MemoryDatas[16],MemoryDatas[17],MemoryDatas[18],MemoryDatas[19]} <= 32'b101;
		{MemoryDatas[20],MemoryDatas[21],MemoryDatas[22],MemoryDatas[23]} <= 32'b110;
		{MemoryDatas[24],MemoryDatas[25],MemoryDatas[26],MemoryDatas[27]} <= 32'b111;
		{MemoryDatas[28],MemoryDatas[29],MemoryDatas[30],MemoryDatas[31]} <= 32'b1000;
		{MemoryDatas[32],MemoryDatas[33],MemoryDatas[34],MemoryDatas[35]} <= 32'b1001;
		{MemoryDatas[36],MemoryDatas[37],MemoryDatas[38],MemoryDatas[39]} <= 32'b1010;
		//last memory
		{MemoryDatas[40],MemoryDatas[41],MemoryDatas[42],MemoryDatas[43]} <= 32'b0;
	end
	//
	if(MemWrite)
		begin
		{MemoryDatas[Address],MemoryDatas[Address+1],MemoryDatas[Address+2],MemoryDatas[Address+3]} <= WriteData;
		//
		end
	end
	
	assign ReadData = (MemRead) ? {MemoryDatas[Address],MemoryDatas[Address+1],MemoryDatas[Address+2],MemoryDatas[Address+3]}: ReadData;
		


endmodule
