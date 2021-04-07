`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   01:42:26 05/20/2018
// Design Name:   TwoToTenConvertor
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 6- working more on behavioral model and state diagram/Matin_Noohnezhad_Assignment4/Matin_Noohnezhad_Assignment4/TestConvertor.v
// Project Name:  Matin_Noohnezhad_Assignment4
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: TwoToTenConvertor
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestConvertor;

	// Inputs
	reg [31:0] bin;
	reg start;
	reg clock;

	// Outputs
	wire [31:0] bcd;
	wire End;

	// Instantiate the Unit Under Test (UUT)
	TwoToTenConvertor uut (
		.bin(bin), 
		.start(start), 
		.clock(clock), 
		.bcd(bcd), 
		.End(End)
	);

	initial begin
		// Initialize Inputs
		bin = 32'b1101000100;
		start = 1;
		clock = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		repeat(100)
		#10 clock = ~clock;
		
	end
      
endmodule

