`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   18:15:35 05/04/2018
// Design Name:   parity_calc
// Module Name:   D:/term 5/Computer Architecture Laboratory/Week 4- Behavioral Modeling/Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/parity_calc_test.v
// Project Name:  Matin_Noohnezhad_Assignment3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: parity_calc
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module parity_calc_test;

	// Inputs
	reg [31:0] d;

	// Outputs
	wire parity;

	// Instantiate the Unit Under Test (UUT)
	parity_calc uut (
		.d(d), 
		.parity(parity)
	);

	initial begin
		// Initialize Inputs
		d = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		d= 32'b010111010110;
		#100;
		d= 32'b000000000000;
		#100;
		d= 32'b000000001100;
		#100;
	end
      
endmodule

