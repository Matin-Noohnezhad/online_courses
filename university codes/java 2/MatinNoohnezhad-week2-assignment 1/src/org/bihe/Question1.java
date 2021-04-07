package org.bihe;

public class Question1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// Next time I put more comment in it.this week i didn't have enough
		// time to make it clean because i was a little sick.It will be better
		// next time.I hope!
		String sentence = "In the Olympic competition which held in Rio in 2016, IRR gained 8 medals. 3 gold medals gained by Kianoosh Rostami, Sohrab Moradi and Hassan Yazdani, 1silver medal gained by Komeil Ghasemi, 4 Bronze medals gained by Kimia Alizadeh, Hassan Rahimi, Saeid Abdevali and Gholamreza Rezaei. In Rio2016 ranking IRR placed in 25. ";
		String[] space = sentence.split(" ");
		String[] comma = sentence.split(",");
		String[] SpaCom = comma[0].split(" ");
		String[] in = sentence.split(" in ");
		String[] country = comma[1].split(" gained ");
		String[] medalCounts = country[1].split(" medals");
		String[] dot = sentence.split("\\.");
		String[] SpaDot = dot[dot.length - 2].split(" ");
		String[] GoldMedal = sentence.split("gold medal");
		String[] SpaGol = GoldMedal[0].split(" ");
		String[] SilverMedal = sentence.split("silver medal");
		String[] SpaSil = SilverMedal[0].split(" ");
		String[] BronzeMedal = sentence.split("ronze medal");
		String[] SpaBrz = BronzeMedal[0].split(" ");
		String[] by = sentence.split("by");
		System.out.println("Competition : " + space[2]);
		System.out.println("Subject : " + in[1] + SpaCom[SpaCom.length - 1]);
		System.out.println("Country :" + country[0]);
		System.out.println("Total Medals : " + medalCounts[0]);
		System.out.println("Country place in table : " + SpaDot[SpaDot.length - 1]);
		System.out.println("---------------------------------------");
		int BySplitNumber = 1;
		System.out.print("Gold Medals : ");
		if (GoldMedal.length != 1) {
			int goldCount = Integer.parseInt(SpaGol[SpaGol.length - 1]);
			String[] SpaBy = by[BySplitNumber].split(" ");
			if (goldCount == 1) {
				System.out.print(SpaBy[1]);
			}
			if (goldCount > 1) {
				String[] and = by[BySplitNumber].split("and ");
				String[] SpaAnd = and[1].split(" ");
				System.out.print(SpaBy[1] + ", ");
				int moreThan2 = 2;
				while (goldCount > moreThan2) {
					moreThan2 += 1;
					String[] commaAfterGold = GoldMedal[1].split(", ");
					String[] SpaCAG = commaAfterGold[moreThan2 - 2].split(" ");
					System.out.print(SpaCAG[0] + ", ");
				}
				System.out.print(SpaAnd[0]);
			}
			BySplitNumber += 1;
		}
		System.out.println();
		System.out.println("---------------------------------------------");
		System.out.print("Silver Medals : ");
		if (SilverMedal.length != 1) {
			int silverCount = Integer.parseInt(SpaSil[SpaSil.length - 1]);
			String[] SpaBy2 = by[BySplitNumber].split(" ");
			if (silverCount == 1) {
				System.out.print(SpaBy2[1]);
			}
			if (silverCount > 1) {
				String[] and = by[BySplitNumber].split("and ");
				String[] SpaAnd = and[1].split(" ");
				System.out.print(SpaBy2[1] + ", ");
				int moreThan2 = 2;
				while (silverCount > moreThan2) {
					moreThan2 += 1;
					String[] commaAfterSilver = SilverMedal[1].split(", ");
					String[] SpaCAS = commaAfterSilver[moreThan2 - 2].split(" ");
					System.out.print(SpaCAS[0] + ", ");
				}
				System.out.print(SpaAnd[0]);
			}
			BySplitNumber += 1;
		}
		System.out.println();
		System.out.println("---------------------------------------------");
		System.out.print("Bronze Medals : ");
		if (BronzeMedal.length != 1) {
			int bronzeCount = Integer.parseInt(SpaBrz[SpaBrz.length - 2]);
			String[] SpaBy3 = by[BySplitNumber].split(" ");
			if (bronzeCount == 1) {
				System.out.print(SpaBy3[1]);
			}
			if (bronzeCount > 1) {
				String[] and = by[BySplitNumber].split("and ");
				String[] SpaAnd = and[1].split(" ");
				System.out.print(SpaBy3[1] + ", ");
				int moreThan2 = 2;
				while (bronzeCount > moreThan2) {
					moreThan2 += 1;
					String[] commaAfterBronze = BronzeMedal[1].split(", ");
					String[] SpaCAB = commaAfterBronze[moreThan2 - 2].split(" ");
					System.out.print(SpaCAB[0] + ", ");
				}
				System.out.print(SpaAnd[0]);
			}
			BySplitNumber += 1;
		}
		System.out.println();
	}

}
