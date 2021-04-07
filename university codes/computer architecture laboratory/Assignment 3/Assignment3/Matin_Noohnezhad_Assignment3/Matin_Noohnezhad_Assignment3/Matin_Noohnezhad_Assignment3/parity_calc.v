`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:09:04 05/04/2018 
// Design Name: 
// Module Name:    parity_calc 
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
module parity_calc #(parameter N=32) (d,parity);
input [N-1:0] d;
output parity;
reg parity;
integer i;

always @(d)begin
	parity = 0;
	for(i=0;i<N;i=i+1)begin
		if(d[i]) parity = ~parity;
	end
end


endmodule
