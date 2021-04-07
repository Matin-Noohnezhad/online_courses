package org.bihe.beans;

public class CommercialEmployee extends Employee {
	private float rewardPercentForEachProject;

	// -----------------------------------------------
	// costructor
	public CommercialEmployee(String firstName, String lastName) {
		super(firstName, lastName);
		this.rewardPercentForEachProject = 0.25f;
		super.incomeInHour = 12000.0f;
		super.insurancePercent = 23;
		super.taxPercent = 2;
		super.extraHoursMultipledBy = 1;
		super.floorOfMoneyForTax = 2000000.0f;
	}

	// ----------------------------------------
	//accessors and mutators
	public float getRewardPercentForEachProject() {
		return rewardPercentForEachProject;
	}
	public void setRewardPercentForEachProject(float rewardPercentForEachProject) {
		this.rewardPercentForEachProject = rewardPercentForEachProject;
	}
	//-------------------------------------------------
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
