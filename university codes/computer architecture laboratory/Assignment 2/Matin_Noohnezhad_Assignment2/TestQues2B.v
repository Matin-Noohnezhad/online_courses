`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   00:07:22 04/27/2018
// Design Name:   Question2B
// Module Name:   D:/XilinxWorkspace/Matin_Noohnezhad_Assignment2/TestQues2B.v
// Project Name:  Matin_Noohnezhad_Assignment2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Question2B
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestQues2B;

	// Inputs
	reg A;
	reg B;
	reg cin;

	// Outputs
	wire S;
	wire cout;

	// Instantiate the Unit Under Test (UUT)
	Question2B uut (
		.A(A), 
		.B(B), 
		.cin(cin), 
		.S(S), 
		.cout(cout)
	);

	initial begin
		// Initialize Inputs
		A = 0;
		B = 0;
		cin = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		A = 1;
		B = 0;
		cin = 0;
		
		#100;
		
		A = 1;
		B = 1;
		cin = 1;
		
		#100;
		
		A = 1;
		B = 0;
		cin = 1;

	end
      
endmodule

