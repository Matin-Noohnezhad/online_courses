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
wire [31:0]insout;
wire [31:0]signExtendOutOut;
wire [1:0]forward_1;
wire [1:0]forward_2;
wire pcWrite;
wire IFIDwrite;
wire bubbleSignal;
wire [31:0]outMux3_1;
wire [31:0]outMux3_2;
wire [31:0]addout;
wire [31:0]IFIDaddOutOut;
wire regdstoutOut;
wire jumpoutOut;
wire branchoutOut;
wire memreadoutOut;
wire memtoregoutOut;
wire memwriteoutOut;
wire alusrcoutOut;
wire regwriteoutOut;
wire [1:0]aluopoutOut;
wire [31:0]readData1Out;
wire [31:0]readData2Out;
wire [4:0]ins2016Out;
wire [4:0]ins1511Out;
wire [4:0]ins2521Out;
wire regDstOut;
wire jumpOut;
wire branchOut;
wire memReadOut;
wire memToRegOut;
wire memWriteOut;
wire aluSrcOut;
wire regWriteOut;
wire [1:0]aluOpOut;

wire [31:0]aluResultOut;
wire zeroOut;
//wire jumpOut;
//wire branchOut;
//wire memReadOut;
//wire memToRegOut;
//wire memWriteOut;
//wire regWriteOut;
wire [31:0]addOut2Out;
wire [31:0]rd2Out;
wire [4:0]muxOut1Out;
wire jumpOutOut;
wire branchOutOut;
wire memToRegOutOut;
wire regWriteOutOut;
wire [31:0]readDataMemoryOut;
wire [31:0]aluResultOutOut;
wire [4:0]muxOut1OutOut;
wire zeroOutOut;

//assign addOut1 = 32'b0;
//assign pcWrite = 0;


PC pc(pcout, addOut1,pcWrite, clk, startin);
Add add_1(pcout, 32'b00000000000000000000000000000100, addOut1);
InstructionMemory instructionMemory(pcout, Ins, startin);
IFID ifid(addOut1,Ins,IFIDwrite,addout,insout,clk);
Mux_5 mux5(ins2016Out,ins1511Out,regdstoutOut,muxOut1);
registerFile registerfile(insout[25:21],insout[20:16], muxOut1, muxOut3, regWriteOutOut, rd1, rd2, startin, clk, val, regNo);
SignExtend signExtend(insout[15:0], sxOut);
Control control(insout[31:26], regDst, jump, branch, memRead, memToReg, aluOp, memWrite, aluSrc, regWrite);
//Mux18to9 mux18to9(regDst,jump,branch,memRead,memToReg,memWrite,aluSrc,regWrite,aluOp,bubbleSignal,regDstOut,jumpOut,branchOut,memReadOut,memToRegOut,memWriteOut,aluSrcOut,regWriteOut,aluOpOut);
IDEX idex(addout,regDst,jump,branch,memRead,memToReg,memWrite,aluSrc,regWrite,aluOp,rd1,rd2,sxOut,insout[20:16],insout[15:11],insout[25:21],clk,IFIDaddOutOut,regdstoutOut,jumpoutOut,branchoutOut,memreadoutOut,memtoregoutOut,memwriteoutOut,alusrcoutOut,regwriteoutOut,aluopoutOut,readData1Out,readData2Out,signExtendOutOut,ins2016Out,ins1511Out,ins2521Out,bubbleSignal);
Mux_32 mux32_1(readData2Out, signExtendOutOut, alusrcoutOut, muxOut2);
ALUControl aluControl(signExtendOutOut[5:0], aluopoutOut, aControl);
//fix the two line below later that you build that 4 registers
Mux3to1 mux3to1_1(readData1Out,aluResultOut,muxOut3,forward_1,outMux3_1);
Mux3to1 mux3to1_2(muxOut2,aluResultOut,muxOut3,forward_2,outMux3_2);
ALU alu(outMux3_1 ,outMux3_2, aControl, aluResult, zero);
EXMEM exmem(aluResult, zero, jumpoutOut,branchoutOut,memreadoutOut,memtoregoutOut, memwriteoutOut,regwriteoutOut, addOut2, readData2Out,muxOut1,clk, aluResultOut, zeroOut, jumpOut, branchOut, memReadOut, memToRegOut, memWriteOut, regWriteOut, addOut2Out, rd2Out,muxOut1Out);
DataMemory dataMemory(aluResultOut, rd2Out, memWriteOut, memReadOut, rd3, startin, clk);
MEMWB memwb(jumpOut, branchOut, memToRegOut, regWriteOut, rd3, aluResultOut, muxOut1Out,zeroOut,clk,jumpOutOut, branchOutOut, memToRegOutOut, regWriteOutOut, readDataMemoryOut, aluResultOutOut, muxOut1OutOut,zeroOutOut);
Mux_32 mux32_2(readDataMemoryOut, aluResultOutOut, ~memToRegOutOut, muxOut3);
ForwardingUnit forwardingUnit(regWriteOut,muxOut1Out,ins2016Out,ins2521Out,regWriteOutOut,muxOut1OutOut,forward_1,forward_2);
HDU hdu(insout[25:21],insout[20:16],memreadoutOut,ins2016Out,pcWrite,IFIDwrite,bubbleSignal);

//Shift_26_28 shift2628(insout[25:0], shiftOut1);
//ShiftLeft2 shiftLeft2(sxOut, shiftOut2);
//Add add_2(addout, shiftOut2, addOut2);
//and(BranchAndZero,branch,zero);
//Mux_32 mux32_3(addout, addOut2, BranchAndZero , muxOut4);
//Mux_32 mux32_4({addout[31:28],shiftOut1}, muxOut4, ~jump, muxOut5);


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
