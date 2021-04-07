`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:28:49 07/04/2018
// Design Name:   InstructionMemory
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestInstructionMemory1.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: InstructionMemory
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestInstructionMemory1;

	// Inputs
	reg [31:0] address;
	reg startin;

	// Outputs
	wire [31:0] instruction;

	// Instantiate the Unit Under Test (UUT)
	InstructionMemory uut (
		.address(address), 
		.instruction(instruction), 
		.startin(startin)
	);

	initial begin
		// Initialize Inputs
		address = 0;
		startin = 0;

		// Wait 100 ns for global reset to finish
		#100;
		startin  =1;
		#10;
		startin = 0;
		address = 32'b100;
        
		// Add stimulus here

	end
      
endmodule

