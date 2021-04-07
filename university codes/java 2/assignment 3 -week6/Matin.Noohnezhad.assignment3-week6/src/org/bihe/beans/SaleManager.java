package org.bihe.beans;

public class SaleManager extends Employee {

	private float rewardPercentForEachProject;

	// -----------------------------------------------
	// costructor
	public SaleManager(String firstName, String lastName) {
		super(firstName, lastName);
		this.rewardPercentForEachProject = 0.5f;
		super.incomeInHour = 20000.0f;
		super.insurancePercent = 28;
		super.taxPercent = 3;
		super.extraHoursMultipledBy = 1;
		super.floorOfMoneyForTax = 4000000.0f;
	}

	// ----------------------------------------
	// accessors and mutators
	public float getRewardPercentForEachProject() {
		return rewardPercentForEachProject;
	}

	public void setRewardPercentForEachProject(float rewardPercentForEachProject) {
		this.rewardPercentForEachProject = rewardPercentForEachProject;
	}

	// -------------------------------------------------
	// override methods
	public void calcReward() {
		float reward = 0;
		for (SellDetail s : sales) {
			if (s.getSellDate().getMonth() + 1 == 11) {
				reward += (s.getProject().getPrice()) * this.rewardPercentForEachProject / 100;
			}
		}
		super.rewardPrice = reward;
	}
	// -----------------------------------------------

}
