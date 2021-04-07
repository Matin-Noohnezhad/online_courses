`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:51:53 07/04/2018
// Design Name:   ShiftLeft2
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestShift32.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: ShiftLeft2
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestShift32;

	// Inputs
	reg [31:0] in;

	// Outputs
	wire [31:0] out;

	// Instantiate the Unit Under Test (UUT)
	ShiftLeft2 uut (
		.in(in), 
		.out(out)
	);

	initial begin
		// Initialize Inputs
		in = 0;

		// Wait 100 ns for global reset to finish
		#100;
		in = 32'b10001;
        
		// Add stimulus here

	end
      
endmodule

