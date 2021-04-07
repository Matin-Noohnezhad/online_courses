`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   17:57:24 04/26/2018
// Design Name:   Question1B
// Module Name:   D:/XilinxWorkspace/Matin_Noohnezhad_Assignment2/TestQues1B.v
// Project Name:  Matin_Noohnezhad_Assignment2
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Question1B
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestQues1B;

	// Inputs
	reg [1:0] A;
	reg [1:0] B;

	// Outputs
	wire [3:0] answer;

	// Instantiate the Unit Under Test (UUT)
	Question1B uut (
		.A(A), 
		.B(B), 
		.answer(answer)
	);

	initial begin
		// Initialize Inputs
		A = 0;
		B = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		A = 0;
		B = 3;
		
		#100;
		
		A = 2;
		B = 1;
		
		#100;
		
		A = 2;
		B = 2;
		
		
		
	end
      
endmodule

