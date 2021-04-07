`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:48:37 07/04/2018
// Design Name:   ALU
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestAlu.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: ALU
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestAlu;

	// Inputs
	reg [31:0] in1;
	reg [31:0] in2;
	reg [3:0] operation;

	// Outputs
	wire [31:0] out;
	wire zero;

	// Instantiate the Unit Under Test (UUT)
	ALU uut (
		.in1(in1), 
		.in2(in2), 
		.operation(operation), 
		.out(out), 
		.zero(zero)
	);

	initial begin
		// Initialize Inputs
		in1 = 0;
		in2 = 0;
		operation = 0;

		// Wait 100 ns for global reset to finish
		#100;
		in1 = 32'b110;
		in2 = 32'b001;
		operation = 4'b10;
		#100;
        
		// Add stimulus here

	end
      
endmodule

