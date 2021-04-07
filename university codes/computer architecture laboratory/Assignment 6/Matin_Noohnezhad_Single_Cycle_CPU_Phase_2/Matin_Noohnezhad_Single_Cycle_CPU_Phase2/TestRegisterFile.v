`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   17:59:46 06/01/2018
// Design Name:   registerFile
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 8 - Single Cycle CPU_Phase2/Matin_Noohnezhad_Single_Cycle_CPU_Phase2/TestRegisterFile.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_CPU_Phase2
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
        
		// Add stimulus here
		clk =1;
		startin = 1;
		
		#10;
		clk = 0;
		startin = 0;
		#10;
		
		clk =1;
		RegWrite = 1;
		WriteData = 113;
		WriteReg = 13;
		
		#10;
		clk=0;
		#10;
		
		clk =1;
		WriteData = 116;
		WriteReg = 16;
		
		#10;
		clk=0;
		#10;
		
		RegWrite = 0 ;
		
		#10;
		clk =1;
		Read1 = 13;
		Read2 = 16;
		regNo = 13;

	end
      
endmodule

