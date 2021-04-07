`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:41:31 05/04/2018 
// Design Name: 
// Module Name:    ones_count 
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
module ones_count #(parameter N=32) (d,count);
input [N-1:0] d;
output [5:0] count;
reg [5:0]count ;
integer i;

always @(d) begin
	count = 6'b0;
	for(i=0;i<N;i=i+1)begin
		if(d[i]) count = count+1;
	end
end


endmodule




