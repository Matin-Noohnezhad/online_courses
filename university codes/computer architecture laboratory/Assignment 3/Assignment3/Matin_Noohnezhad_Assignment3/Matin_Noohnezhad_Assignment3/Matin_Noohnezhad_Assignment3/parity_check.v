`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:22:51 05/04/2018 
// Design Name: 
// Module Name:    parity_check 
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
module parity_check #(parameter N=32) (d,parity,err);
input [N-1:0] d;
input parity;
output err;
reg err;
integer i;

always @(*)
begin
	err = 0;
	for(i=0;i<N;i=i+1)begin
		if(d[i]) err = ~err;
	end
	if(parity) err = ~err;
end


endmodule
