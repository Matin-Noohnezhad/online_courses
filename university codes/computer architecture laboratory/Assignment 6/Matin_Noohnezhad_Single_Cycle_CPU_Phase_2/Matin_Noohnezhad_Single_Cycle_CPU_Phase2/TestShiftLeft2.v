`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   16:16:19 06/01/2018
// Design Name:   ShiftLeft2
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 8 - Single Cycle CPU_Phase2/Matin_Noohnezhad_Single_Cycle_CPU_Phase2/TestShiftLeft2.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_CPU_Phase2
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

module TestShiftLeft2;

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
        
		// Add stimulus here
		in = 34;
		
		

	end
      
endmodule

