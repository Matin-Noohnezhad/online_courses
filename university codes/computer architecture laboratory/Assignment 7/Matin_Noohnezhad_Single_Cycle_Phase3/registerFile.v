`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:19:28 06/01/2018 
// Design Name: 
// Module Name:    registerFile 
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
module registerFile(Read1, Read2, WriteReg, WriteData, RegWrite, Data1, Data2, startin, clk, val, regNo);

input [4:0]Read1;
input [4:0]Read2;
input [4:0]WriteReg;
input [31:0]WriteData;
input RegWrite;
input startin;
input clk;
input [4:0]regNo;
output [31:0]Data1;
output [31:0]Data2;
output [31:0] val;
reg [31:0] register [31:0];

always @(posedge clk)
	begin
	//do the write operation
	if(startin) 
	begin
		register[0] <= 32'b0;
		register[1] <= 32'b0;
		register[2] <= 32'b0;
		register[3] <= 32'b0;
		register[4] <= 32'b0;
		register[5] <= 32'b0;
		register[6] <= 32'b0;
		register[7] <= 32'b0;
		register[8] <= 32'b0;
		register[9] <= 32'b0;
		register[10] <= 32'b0;
		register[11] <= 32'b0;
		register[12] <= 32'b0;
		register[13] <= 32'b0;
		register[14] <= 32'b0;
		register[15] <= 32'b0;
		register[16] <= 32'b0;
		register[17] <= 32'b0;
		register[18] <= 32'b0;
		register[19] <= 32'b0;
		register[20] <= 32'b0;
		register[21] <= 32'b0;
		register[22] <= 32'b0;
		register[23] <= 32'b0;
		register[24] <= 32'b0;
		register[25] <= 32'b0;
		register[26] <= 32'b0;
		register[27] <= 32'b0;
		register[28] <= 32'b0;
		register[29] <= 32'b0;
		register[30] <= 32'b0;
		register[31] <= 32'b0;
		register[32] <= 32'b0;
		end
	if(RegWrite)
	if(WriteReg !=0)
			register[WriteReg] <= WriteData;
	//do the write operation
	end
	assign Data1 = register[Read1];
	assign Data2 = register[Read2];
	assign val = register[regNo];
endmodule
