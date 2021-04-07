`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   23:37:36 07/12/2018
// Design Name:   Main
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 13- Pipeline CPU Project/Pipeline_CPU_Phase1/TestMain.v
// Project Name:  Pipeline_CPU_Phase1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Main
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestMain;

	// Inputs
	reg clk;
	reg startin;
	reg [4:0] regNo;

	// Outputs
	wire [31:0] val;

	// Instantiate the Unit Under Test (UUT)
	Main uut (
		.clk(clk), 
		.startin(startin), 
		.regNo(regNo), 
		.val(val)
	);

	initial begin
		// Initialize Inputs
		clk = 0;
		startin = 0;
		regNo = 0;

		// Wait 100 ns for global reset to finish
		#10;
        
		// Add stimulus here
		startin = 1;
		clk = 1;
		regNo = 5'b10001;
		#10;
		startin = 0;
		repeat(50)
		#10 clk = ~clk;

	end
      
endmodule

