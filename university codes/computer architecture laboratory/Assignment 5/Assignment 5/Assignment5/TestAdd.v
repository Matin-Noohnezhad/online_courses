`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   23:21:15 05/25/2018
// Design Name:   Add
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 7- Single Cycle CPU_ Phase1/Assignment 5/Assignment5/TestAdd.v
// Project Name:  Assignment5
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

module TestAdd;

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
        
		// Add stimulus here
		digit1 = 32'b0010;
		digit2 = 32'b0110;
		

	end
      
endmodule

