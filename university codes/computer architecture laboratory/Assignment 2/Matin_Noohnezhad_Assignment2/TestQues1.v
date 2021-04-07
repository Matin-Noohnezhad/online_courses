`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   17:30:40 04/26/2018
// Design Name:   Question1
// Module Name:   D:/XilinxWorkspace/Matin_Noohnezhad_Assignment2/TestQues1.v
// Project Name:  Matin_Noohnezhad_Assignment2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Question1
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestQues1;

	// Inputs
	reg A;
	reg B;

	// Outputs
	wire sum;
	wire cout;

	// Instantiate the Unit Under Test (UUT)
	Question1 uut (
		.A(A), 
		.B(B), 
		.sum(sum), 
		.cout(cout)
	);

	initial begin
		// Initialize Inputs
		A = 0;
		B = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		A = 1;
		B = 1;
		
		#100;
		
		A = 1;
		B = 0;
		
		#100;
		
		A = 0;
		B = 1;
		
		#100;
		
		A = 0;
		B = 0;
		
		
	end
      
endmodule

