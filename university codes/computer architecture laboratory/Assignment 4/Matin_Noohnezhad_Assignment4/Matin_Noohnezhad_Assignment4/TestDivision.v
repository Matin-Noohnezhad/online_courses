`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   03:30:09 05/20/2018
// Design Name:   Division
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 6- working more on behavioral model and state diagram/Matin_Noohnezhad_Assignment4/Matin_Noohnezhad_Assignment4/TestDivision.v
// Project Name:  Matin_Noohnezhad_Assignment4
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Division
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestDivision;

	// Inputs
	reg [7:0] dividend;
	reg [7:0] divisor;
	reg clock;

	// Outputs
	wire [7:0] quotient;
	wire [7:0] remainder;
	wire [7:0] tempDivisor;

	// Instantiate the Unit Under Test (UUT)
	Division uut (
		.dividend(dividend), 
		.divisor(divisor), 
		.quotient(quotient), 
		.remainder(remainder), 
		.tempDivisor(tempDivisor), 
		.clock(clock)
	);

	initial begin
		// Initialize Inputs
		dividend = 7;
		divisor = 2;
		clock = 0;

		// Wait 100 ns for global reset to finish
		#100;
        
		// Add stimulus here
		
		repeat (60)
		#10 clock = ~clock;

	end
      
endmodule

