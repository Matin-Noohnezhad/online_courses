`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   02:42:01 07/04/2018
// Design Name:   Control
// Module Name:   D:/term 5/Computer Architecture Laboratory/week 12 - Single Cycle CPU_Phase3/Matin_Noohnezhad_Single_Cycle_Phase3/TestControl.v
// Project Name:  Matin_Noohnezhad_Single_Cycle_Phase3
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Control
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module TestControl;

	// Inputs
	reg [5:0] instruction31_26;

	// Outputs
	wire regdst;
	wire jump;
	wire branch;
	wire memread;
	wire memtoreg;
	wire [1:0] aluop;
	wire memwrite;
	wire alusrc;
	wire regwrite;

	// Instantiate the Unit Under Test (UUT)
	Control uut (
		.instruction31_26(instruction31_26), 
		.regdst(regdst), 
		.jump(jump), 
		.branch(branch), 
		.memread(memread), 
		.memtoreg(memtoreg), 
		.aluop(aluop), 
		.memwrite(memwrite), 
		.alusrc(alusrc), 
		.regwrite(regwrite)
	);

	initial begin
		// Initialize Inputs
		instruction31_26 = 0;

		// Wait 100 ns for global reset to finish
		#100;
        instruction31_26 = 6'b100011;
		  #100;
		  instruction31_26 = 6'b0;
		  #100;
		  instruction31_26 = 6'b101011;
		  
		// Add stimulus here

	end
      
endmodule

