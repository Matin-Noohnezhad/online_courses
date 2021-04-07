package org.bihe.beans;

public class TechnicalManager extends Employee {

	// costructor
	public TechnicalManager(String firstName, String lastName) {
		super(firstName, lastName);
		super.incomeInHour = 20000.0f;
		super.insurancePercent = 28;
		super.taxPercent = 3;
		super.extraHoursMultipledBy = 1.5f;
		super.floorOfMoneyForTax = 4000000.0f;
	}

	// ----------------------------------------
	// override methods
	public void calcReward() {
		float reward = 0;
		super.rewardPrice = reward;
	}
	// -----------------------------------------------

}
