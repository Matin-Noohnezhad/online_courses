`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:52:56 07/04/2018
// Design Name:   DataMemory
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestDataMemory.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
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
		MemRead = 1;
		MemWrite = 0;
		Address = 32'b1000;
		startin = 1;
		clk = 1;
		WriteData = 32'b101;
        #10;
		  startin = 0;
		  clk = 0;
		  #10;
		  clk =1;
		  #10;
		  clk = 0;
		  #10;
		  MemRead = 0;
		  MemWrite = 1;
		  clk = 1;
		  
		// Add stimulus here

	end
      
endmodule

