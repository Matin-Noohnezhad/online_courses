`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:31:16 07/04/2018
// Design Name:   Mux_5
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestMux5.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Mux_5
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestMux5;

	// Inputs
	reg [4:0] input1;
	reg [4:0] input2;
	reg op;

	// Outputs
	wire [4:0] out;

	// Instantiate the Unit Under Test (UUT)
	Mux_5 uut (
		.input1(input1), 
		.input2(input2), 
		.op(op), 
		.out(out)
	);

	initial begin
		// Initialize Inputs
		input1 = 0;
		input2 = 0;
		op = 0;

		// Wait 100 ns for global reset to finish
		#100;
		input1 = 5'b100;
		input2 = 5'b111;
		op = 1;
        
		// Add stimulus here

	end
      
endmodule

