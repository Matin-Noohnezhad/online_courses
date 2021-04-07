`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:56:23 05/17/2018 
// Design Name: 
// Module Name:    TwoToTenConvertor 
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
module TwoToTenConvertor(bin,,start,clock,bcd,End);
input [31:0]bin;
input start;
input clock;
output [31:0]bcd;
output End;
reg [31:0]bcd;
reg End;
integer counter;
reg [31:0]tempBin;
reg addToBlockStatus;

initial begin
bcd = 32'b0;
End = 0;
counter = 0;
addToBlockStatus = 0;
end

always @(posedge clock)
begin
if(start & ~End & counter<32)
begin
if(counter == 0)
tempBin = bin;
if(addToBlockStatus|(bcd[31:28]<4'b101&bcd[27:24]<4'b101&bcd[23:20]<4'b101&bcd[19:16]<4'b101&bcd[15:12]<4'b101&bcd[11:8]<4'b101&bcd[7:4]<4'b101&bcd[3:0]<4'b101))
begin
addToBlockStatus <= 0;
$display("1 if");
if(tempBin[31])begin
$display("2 if");
bcd <= (bcd << 1)+ 32'b1;
end
else
begin
$display("3 if");
bcd <= (bcd << 1) + 32'b0;
end
$display("4 if");
tempBin <= (tempBin << 1) + 32'b0;
if(counter == 32) 
End <= 1;

counter = counter + 1;
end
else begin
if(bcd[31:28]>4'b100)begin
bcd[31:28] <= bcd[31:28] + 4'b011;
end
if(bcd[27:24]>4'b100)begin
bcd[27:24] <= bcd[27:24] + 4'b011;
end
if(bcd[23:20]>4'b100)begin
bcd[23:20] <= bcd[23:20] + 4'b011;
end
if(bcd[19:16]>4'b100)begin
bcd[19:16] <= bcd[19:16] + 4'b011;
end
if(bcd[15:12]>4'b100)begin
bcd[15:12] <= bcd[15:12] + 4'b011;
end
if(bcd[11:8]>4'b100)begin
bcd[11:8] <= bcd[11:8] + 4'b011;
end
if(bcd[7:4]>4'b100)begin
bcd[7:4] <= bcd[7:4] + 4'b011;
end
if(bcd[3:0]>4'b100)begin
bcd[3:0] <= bcd[3:0] + 4'b011;
end
addToBlockStatus <= 1;
end
end
end

endmodule
