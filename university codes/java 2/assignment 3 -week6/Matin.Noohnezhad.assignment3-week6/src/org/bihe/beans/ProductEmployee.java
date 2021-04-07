package org.bihe.beans;

public class ProductEmployee extends Employee {

	// costructor
	public ProductEmployee(String firstName, String lastName) {
		super(firstName, lastName);
		super.incomeInHour = 12000.0f;
		super.insurancePercent = 23;
		super.taxPercent = 2;
		super.extraHoursMultipledBy = 1.5f;
		super.floorOfMoneyForTax = 2000000.0f;
	}

	// ----------------------------------------
	// override methods
	public void calcReward() {
		float reward = 0;
		super.rewardPrice = reward;
	}
	// -----------------------------------------------

}
