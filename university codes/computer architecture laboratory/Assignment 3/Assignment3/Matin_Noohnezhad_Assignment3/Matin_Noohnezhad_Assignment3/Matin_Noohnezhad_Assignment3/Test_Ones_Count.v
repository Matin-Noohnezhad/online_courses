`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   17:47:21 05/04/2018
// Design Name:   ones_count
// Module Name:   D:/term 5/Computer Architecture Laboratory/Week 4- Behavioral Modeling/Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/Matin_Noohnezhad_Assignment3/Test_Ones_Count.v
// Project Name:  Matin_Noohnezhad_Assignment3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: ones_count
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module Test_Ones_Count;

	// Inputs
	reg [31:0] d;

	// Outputs
	wire [5:0] count;

	// Instantiate the Unit Under Test (UUT)
	ones_count uut (
		.d(d), 
		.count(count)
	);

	initial begin
		// Initialize Inputs
		d = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		d = 32'b01001100;
		#100;
		d = 32'b0;
		#100;
		d = 32'b11;
		#100;

	end
      
endmodule

