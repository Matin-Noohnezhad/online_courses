`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:47:10 07/04/2018
// Design Name:   ALUControl
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestAluControl.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: ALUControl
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestAluControl;

	// Inputs
	reg [5:0] Func;
	reg [1:0] Aluop;

	// Outputs
	wire [3:0] Alucontrol;

	// Instantiate the Unit Under Test (UUT)
	ALUControl uut (
		.Func(Func), 
		.Aluop(Aluop), 
		.Alucontrol(Alucontrol)
	);

	initial begin
		// Initialize Inputs
		Func = 0;
		Aluop = 0;

		// Wait 100 ns for global reset to finish
		#100;
		
        
		// Add stimulus here

	end
      
endmodule

