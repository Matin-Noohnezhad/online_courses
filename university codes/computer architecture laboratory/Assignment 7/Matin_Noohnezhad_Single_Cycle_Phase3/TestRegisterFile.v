`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:33:18 07/04/2018
// Design Name:   registerFile
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestRegisterFile.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: registerFile
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestRegisterFile;

	// Inputs
	reg [4:0] Read1;
	reg [4:0] Read2;
	reg [4:0] WriteReg;
	reg [31:0] WriteData;
	reg RegWrite;
	reg startin;
	reg clk;
	reg [4:0] regNo;

	// Outputs
	wire [31:0] Data1;
	wire [31:0] Data2;
	wire [31:0] val;

	// Instantiate the Unit Under Test (UUT)
	registerFile uut (
		.Read1(Read1), 
		.Read2(Read2), 
		.WriteReg(WriteReg), 
		.WriteData(WriteData), 
		.RegWrite(RegWrite), 
		.Data1(Data1), 
		.Data2(Data2), 
		.startin(startin), 
		.clk(clk), 
		.val(val), 
		.regNo(regNo)
	);

	initial begin
		// Initialize Inputs
		Read1 = 0;
		Read2 = 0;
		WriteReg = 0;
		WriteData = 0;
		RegWrite = 0;
		startin = 0;
		clk = 0;
		regNo = 0;

		// Wait 100 ns for global reset to finish
		#100;
		startin = 1;
		clk = 1;
        
		  #10;
		// Add stimulus here
startin = 0;
WriteReg = 5'b101;
WriteData = 32'b1101;
RegWrite = 1;		
Read1 =   5'b0;
Read2 =   5'b0;


clk = 0;

#10;
clk = 1;
Read1 =   5'b101;
Read2 =   5'b101;

	end
      
endmodule

