`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:21:57 07/04/2018
// Design Name:   PC
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestPc1.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
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

module TestPc1;

	// Inputs
	reg [31:0] in;
	reg clk;
	reg startin;

	// Outputs
	wire [31:0] out;

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
		clk = 1;
		startin = 1;
		#10;
		startin = 0;
		in = 32'b100;
		repeat(10)
		#10 clk = ~clk;
        
		// Add stimulus here

	end
      
endmodule

