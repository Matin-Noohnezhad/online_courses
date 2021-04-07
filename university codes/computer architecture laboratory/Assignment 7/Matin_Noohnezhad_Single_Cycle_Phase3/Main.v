`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    23:08:38 07/01/2018 
// Design Name: 
// Module Name:    Main 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Main(clk, startin, regNo, val);

input clk;
input startin;
input [4:0]regNo;
output [31:0]val;
//
wire [4:0]regNo;
wire [31:0]val;
wire [31:0]pcout;
wire [31:0]Ins;
//wire [5:0]Ins3126;
//wire [4:0]Ins2521;
//wire [4:0]Ins2016;
//wire [4:0]Ins1511;
//wire [15:0]Ins150;
//wire [5:0]Ins50;
//wire [25:0]Ins250;
wire [4:0]muxOut1;
wire [31:0]rd1;
wire [31:0]rd2;
wire [31:0]muxOut2;
wire zero;
wire [31:0]aluResult;
wire [31:0]rd3;
wire [31:0]muxOut3;
wire [31:0]sxOut;
wire [3:0]aControl;
wire regDst;
wire jump;
wire branch;
wire memRead;
wire memToReg;
wire [1:0]aluOp;
wire memWrite;
wire aluSrc;
wire regWrite;
wire [31:0]addOut1;
wire [27:0]shiftOut1;
wire [31:0]shiftOut2;
wire [31:0]addOut2;
wire [31:0]muxOut4;
wire [31:0]muxOut5;
wire BranchAndZero;

//assign muxOut5 = 32'b0;

PC pc(pcout, muxOut5, clk, startin);
InstructionMemory instructionMemory(pcout, Ins, startin);
Mux_5 mux5(Ins[20:16],Ins[15:11],regDst,muxOut1);
registerFile registerfile(Ins[25:21],Ins[20:16], muxOut1, muxOut3, regWrite, rd1, rd2, startin, clk, val, regNo);
Mux_32 mux32_1(rd2, sxOut, aluSrc, muxOut2);
ALU alu(rd1 ,muxOut2, aControl, aluResult, zero);
DataMemory dataMemory(aluResult, rd2, memWrite, memRead, rd3, startin, clk);
Mux_32 mux32_2(rd3, aluResult, ~memToReg, muxOut3);
SignExtend signExtend(Ins[15:0], sxOut);
ALUControl aluControl(Ins[5:0], aluOp, aControl);
Control control(Ins[31:26], regDst, jump, branch, memRead, memToReg, aluOp, memWrite, aluSrc, regWrite);
Add add_1(pcout, 32'b00000000000000000000000000000100, addOut1);
Shift_26_28 shift2628(Ins[25:0], shiftOut1);
ShiftLeft2 shiftLeft2(sxOut, shiftOut2);
Add add_2(addOut1, shiftOut2, addOut2);
and(BranchAndZero,branch,zero);
Mux_32 mux32_3(addOut1, addOut2, BranchAndZero , muxOut4);
Mux_32 mux32(shiftOut1, muxOut4, ~jump, muxOut5);

//
//modules
//ALU alu(in1, in2, operation, out, zero);
//ALUControl aluControl(Func, Aluop, Alucontrol);
//Add add(digit1, digit2, result);
//Control control(instruction31_26, regdst, jump, branch, memread, memtoreg, aluop, memwrite, alusrc, regwrite);
//DataMemory dataMemory(Address, WriteData, MemWrite, MemRead, ReadData, startin, clk);
//InstructionMemory instructionMemory(address, instruction, startin);
//Mux_32 mux32(input1, input2, op, out);
//Mux_5 mux5(input1,input2,op,out);
//PC pc(out, in, clk, startin);
//ShiftLeft2 shiftLeft2(in, out);
//Shift_26_28 shift2628(in, out);
//SignExtend signExtend(in, result);
//registerFile registerfile(Read1, Read2, WriteReg, WriteData, RegWrite, Data1, Data2, startin, clk, val, regNo);


endmodule
