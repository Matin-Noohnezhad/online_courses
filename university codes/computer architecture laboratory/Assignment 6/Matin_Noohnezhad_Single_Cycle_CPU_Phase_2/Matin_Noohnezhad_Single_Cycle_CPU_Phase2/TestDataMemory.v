`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   22:48:13 06/01/2018
// Design Name:   DataMemory
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 8 - Single Cycle CPU_Phase2/Matin_Noohnezhad_Single_Cycle_CPU_Phase_2/Matin_Noohnezhad_Single_Cycle_CPU_Phase2/TestDataMemory.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_CPU_Phase2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: DataMemory
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestDataMemory;

	// Inputs
	reg [31:0] Address;
	reg [31:0] WriteData;
	reg MemWrite;
	reg MemRead;
	reg startin;
	reg clk;

	// Outputs
	wire [31:0] ReadData;

	// Instantiate the Unit Under Test (UUT)
	DataMemory uut (
		.Address(Address), 
		.WriteData(WriteData), 
		.MemWrite(MemWrite), 
		.MemRead(MemRead), 
		.ReadData(ReadData), 
		.startin(startin), 
		.clk(clk)
	);

	initial begin
		// Initialize Inputs
		Address = 0;
		WriteData = 0;
		MemWrite = 0;
		MemRead = 0;
		startin = 0;
		clk = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
			clk = 1;
		startin = 1;
		#10;
		clk =1;
		startin = 0;
		Address = 2 ;
		MemRead = 1;
		MemWrite =0;
		 #10;

	end
      
endmodule

