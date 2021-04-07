`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   20:39:49 04/26/2018
// Design Name:   Question2A
// Module Name:   D:/XilinxWorkspace/Matin_Noohnezhad_Assignment2/TestQues2A.v
// Project Name:  Matin_Noohnezhad_Assignment2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Question2A
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestQues2A;

	// Inputs
	reg [2:0] A;

	// Outputs
	wire zero;
	wire one;
	wire two;
	wire three;
	wire four;
	wire five;
	wire six;
	wire seven;

	// Instantiate the Unit Under Test (UUT)
	Question2A uut (
		.A(A), 
		.zero(zero), 
		.one(one), 
		.two(two), 
		.three(three), 
		.four(four), 
		.five(five), 
		.six(six), 
		.seven(seven)
	);

	initial begin
		// Initialize Inputs
		A = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		A = 110;
		#100;
		A = 111;
		#100;
		A = 000;
	end
      
endmodule

