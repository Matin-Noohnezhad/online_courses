`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    02:46:17 07/10/2018 
// Design Name: 
// Module Name:    IFID 
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
module IFID(addin,insin,IFIDwrite,addout,insout,clk);

input [31:0]addin;
input [31:0]insin;
input IFIDwrite;
input clk;
output [31:0]addout;
output [31:0]insout;
reg [31:0]addout;
reg [31:0]insout;


always @(posedge clk)
begin
if(IFIDwrite == 0)
addout <= addin;
insout <= insin;
end





endmodule
