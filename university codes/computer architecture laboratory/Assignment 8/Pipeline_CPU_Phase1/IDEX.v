`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    02:46:41 07/10/2018 
// Design Name: 
// Module Name:    IDEX 
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
module IDEX(IFIDaddOut,regdstout,jumpout,branchout,memreadout,memtoregout,memwriteout,alusrcout,regwriteout,aluopout,readData1,readData2,signExtendOut,ins2016,ins1511,ins2521,clk,IFIDaddOutOut,regdstoutOut,jumpoutOut,branchoutOut,memreadoutOut,memtoregoutOut,memwriteoutOut,alusrcoutOut,regwriteoutOut,aluopoutOut,readData1Out,readData2Out,signExtendOutOut,ins2016Out,ins1511Out,ins2521Out,bubbleSignal);
input [31:0]IFIDaddOut;
input regdstout;
input jumpout;
input branchout;
input memreadout;
input memtoregout;
input memwriteout;
input alusrcout;
input regwriteout;
input [1:0]aluopout;
input [31:0]readData1;
input [31:0]readData2;
input [31:0]signExtendOut;
input [4:0]ins2016;
input [4:0]ins1511; 
input [4:0]ins2521; 
input clk;
input bubbleSignal;
output reg [31:0]IFIDaddOutOut;
output reg regdstoutOut;
output reg jumpoutOut;
output reg branchoutOut;
output reg memreadoutOut;
output reg memtoregoutOut;
output reg memwriteoutOut;
output reg alusrcoutOut;
output reg regwriteoutOut;
output reg [1:0]aluopoutOut;
output reg [31:0]readData1Out;
output reg [31:0]readData2Out;
output reg [31:0]signExtendOutOut;
output reg [4:0]ins2016Out;
output reg [4:0]ins1511Out;
output reg [4:0]ins2521Out;

always @(posedge clk)
begin
IFIDaddOutOut <=IFIDaddOut;
regdstoutOut <=regdstout;
jumpoutOut<=jumpout;
branchoutOut<=branchout;
memreadoutOut<=memreadout;
memtoregoutOut<=memtoregout;
memwriteoutOut<=memwriteout;
alusrcoutOut<=alusrcout;
regwriteoutOut<=regwriteout;
aluopoutOut<=aluopout;
readData1Out<=readData1;
readData2Out<=readData2;
signExtendOutOut<=signExtendOut;
ins2016Out<=ins2016;
ins1511Out<=ins1511;
ins2521Out<=ins2521;

if(bubbleSignal)
begin
regdstoutOut <=0;
jumpoutOut<=0;
branchoutOut<=0;
memreadoutOut<=0;
memtoregoutOut<=0;
memwriteoutOut<=0;
alusrcoutOut<=0;
regwriteoutOut<=0;
aluopoutOut<=2'b0;
end

end

endmodule
