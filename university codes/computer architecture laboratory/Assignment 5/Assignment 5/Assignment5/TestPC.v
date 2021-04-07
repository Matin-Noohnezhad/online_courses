`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   23:42:19 05/25/2018
// Design Name:   PC
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 7- Single Cycle CPU_ Phase1/Assignment 5/Assignment5/TestPC.v
// Project Name:  Assignment5
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: PC
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestPC;

	// Inputs
	reg [7:0] in;
	reg clk;
	reg startin;

	// Outputs
	wire [7:0] out;

	// Instantiate the Unit Under Test (UUT)
	PC uut (
		.out(out), 
		.in(in), 
		.clk(clk), 
		.startin(startin)
	);

	initial begin
		// Initialize Inputs
		in = 0;
		clk = 0;
		startin = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		startin = 0;
		clk= 1;
		in = 8'b1000;

	end
      
endmodule

