`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:54:56 05/25/2018 
// Design Name: 
// Module Name:    PC 
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
module PC(out, in, clk, startin);
output [31:0]out;
input [31:0]in;
input clk;
input startin;
reg [31:0] out;

always@(posedge clk)
    begin
        if(startin == 1)
            begin
                out <= 0;
					 //startin <=0;
            end
        else
            begin
                out <= in;
            end             
    end
endmodule
