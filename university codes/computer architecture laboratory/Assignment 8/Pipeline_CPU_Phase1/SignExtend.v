`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:17:41 06/01/2018 
// Design Name: 
// Module Name:    SignExtend 
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
module SignExtend(in, result);

input [15:0] in;
output [31:0]result;
always@(in)
begin
$display("result",result);
end
assign result = (~in[15])? {16'b0 , in}:{16'b1111111111111111 , in};

endmodule
