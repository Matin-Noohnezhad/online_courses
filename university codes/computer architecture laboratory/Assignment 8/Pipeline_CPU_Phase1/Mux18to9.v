`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:21:08 07/11/2018 
// Design Name: 
// Module Name:    Mux18to9 
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
module Mux18to9(regdst,jump,branch,memread,memtoreg,memwrite,alusrc,regwrite,aluop,bubbleSignal,regdstout,jumpout,branchout,memreadout,memtoregout,memwriteout,alusrcout,regwriteout,aluopout);

input regdst;
input jump;
input branch;
input memread;
input memtoreg;
input memwrite;
input alusrc;
input regwrite;
input [1:0]aluop;

input bubbleSignal;

output regdstout;
output jumpout;
output branchout;
output memreadout;
output memtoregout;
output memwriteout;
output alusrcout;
output regwriteout;
output [1:0]aluopout;


assign regdstout = (bubbleSignal)?0:regdst;
assign jumpout= (bubbleSignal)?0:jump;
assign branchout= (bubbleSignal)?0:branch;
assign memreadout= (bubbleSignal)?0:memread;
assign memtoregout= (bubbleSignal)?0:memtoreg;
assign memwriteout= (bubbleSignal)?0:memwrite;
assign alusrcout= (bubbleSignal)?0:alusrc;
assign regwriteout= (bubbleSignal)?0:regwrite;
assign aluopout= (bubbleSignal)?2'b0:aluop;

always @(memreadout)
begin
$display("memreadout		",memreadout);
$display("jumpout		",jumpout);
$display("branchout		",branchout);
$display("regdstout		",regdstout);
$display("memtoregout		",memtoregout);
$display("memwriteout		",memwriteout);
$display("alusrcout		",alusrcout);
$display("regwriteout		",regwriteout);
$display("aluopout		",aluopout);
end

//always @(bubbleSignal)
//$display("bubbbbbllllle", bubbleSignal);

endmodule
