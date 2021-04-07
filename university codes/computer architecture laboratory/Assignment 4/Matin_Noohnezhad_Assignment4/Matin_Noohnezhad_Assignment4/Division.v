`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:00:50 05/18/2018 
// Design Name: 
// Module Name:    Division 
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
module Division(dividend, divisor, quotient,remainder,tempDivisor,clock);
input wire [7:0]dividend;
input wire [7:0]divisor;
input wire clock;
output [7:0]remainder;
output [7:0]quotient;
//output [7:0]finalQuotient;
//output [7:0]finalRemainder;
output [7:0] tempDivisor;
integer counter;
reg [7:0] tempDivisor;
reg [7:0]remainder;
reg [7:0]quotient;
//reg [7:0]finalQuotient;
//reg [7:0]finalRemainder;
reg remainderChangeStatus;

initial
begin
quotient = 8'b0;
remainder = 8'b0;
counter = 0;
tempDivisor = 8'b11;
remainderChangeStatus = 0;
end


always @(posedge clock)begin
if(tempDivisor > 8'b1)begin
//
if(counter == 0) begin
remainder <= dividend;
tempDivisor <= divisor<<4;
counter <= counter + 1;
end
else
begin
if(~remainderChangeStatus) begin
remainder <= remainder - tempDivisor;
remainderChangeStatus <= 1;
end

if(remainderChangeStatus)begin
if(~remainder[7])begin
quotient <= (quotient << 1) + 8'b1;
end

if(remainder[7]) begin
remainder <= remainder + tempDivisor;
quotient <= (quotient << 1) + 8'b0; 
end

tempDivisor <= tempDivisor >> 1 ;
remainderChangeStatus <= 0;
end

end
//else begin
//finalQuotient <= quotient;
//finalRemainder <= remainder;
//end
//
end
end


endmodule
