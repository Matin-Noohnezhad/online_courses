`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   00:21:06 04/27/2018
// Design Name:   Question3
// Module Name:   D:/XilinxWorkspace/Matin_Noohnezhad_Assignment2/TestQues3.v
// Project Name:  Matin_Noohnezhad_Assignment2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Question3
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestQues3;

	// Inputs
	reg A;
	reg B;
	reg C;
	reg D;

	// Outputs
	wire out;

	// Instantiate the Unit Under Test (UUT)
	Question3 uut (
		.A(A), 
		.B(B), 
		.C(C), 
		.D(D), 
		.out(out)
	);

	initial begin
		// Initialize Inputs
		A = 0;
		B = 0;
		C = 0;
		D = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		A = 1;
		B = 1;
		C = 0;
		D = 0;
		
		#100;
		
		A = 0;
		B = 1;
		C = 1;
		D = 1;
		
		#100;
		
		A = 1;
		B = 1;
		C = 1;
		D = 1;

	end
      
endmodule

