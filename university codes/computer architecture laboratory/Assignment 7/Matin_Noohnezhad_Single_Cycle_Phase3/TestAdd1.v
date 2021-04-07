`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:27:24 07/04/2018
// Design Name:   Add
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestAdd1.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Add
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestAdd1;

	// Inputs
	reg [31:0] digit1;
	reg [31:0] digit2;

	// Outputs
	wire [31:0] result;

	// Instantiate the Unit Under Test (UUT)
	Add uut (
		.digit1(digit1), 
		.digit2(digit2), 
		.result(result)
	);

	initial begin
		// Initialize Inputs
		digit1 = 0;
		digit2 = 0;

		// Wait 100 ns for global reset to finish
		#100;
		digit1 = 32'b10;
		digit2 = 32'b100;
		#10;
		
        
		// Add stimulus here

	end
      
endmodule

