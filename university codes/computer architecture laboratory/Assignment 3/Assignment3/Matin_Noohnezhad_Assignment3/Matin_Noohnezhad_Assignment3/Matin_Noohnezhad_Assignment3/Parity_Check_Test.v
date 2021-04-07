`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   18:33:34 05/04/2018
// Design Name:   parity_check
// Module Name:   D:/term 5/Computer Architecture Laboratory/Week 4- Behavioral Modeling/Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/Parity_Check_Test.v
// Project Name:  Matin_Noohnezhad_Assignment3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: parity_check
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module Parity_Check_Test;

	// Inputs
	reg [31:0] d;
	reg parity;

	// Outputs
	wire err;

	// Instantiate the Unit Under Test (UUT)
	parity_check uut (
		.d(d), 
		.parity(parity), 
		.err(err)
	);

	initial begin
		// Initialize Inputs
		d = 0;
		parity = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		d= 32'b010111010110;
		parity = 1;
		#100;
		d= 32'b010111010110;
		parity = 0;
		#100;
		d= 32'b000000000000;
		parity = 0;
		#100;
		d= 32'b000000000000;
		parity = 1;
		#100;
		d= 32'b000000001100;
		parity = 0;
		#100;
		d= 32'b000000001100;
		parity = 1;
		#100;
		
		
	end
      
endmodule

