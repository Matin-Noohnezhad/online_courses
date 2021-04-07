package org.bihe.beans;

public class CommercialManager extends Employee {

	private float rewardForEachProject;

	// ---------------------------------------
	// costructor
	public CommercialManager(String firstName, String lastName) {
		super(firstName, lastName);
		this.rewardForEachProject = 200000;
		super.incomeInHour = 20000.0f;
		super.insurancePercent = 28;
		super.taxPercent = 3;
		super.extraHoursMultipledBy = 1;
		super.floorOfMoneyForTax = 4000000.0f;
	}

	// ----------------------------------------
	// accessors and mutators
	public float getRewardForEachProject() {
		return rewardForEachProject;
	}

	public void setRewardForEachProject(float rewardForEachProject) {
		this.rewardForEachProject = rewardForEachProject;
	}

	// -------------------------------------------
	// override methods
	public void calcReward() {
		float reward = 0;
		for (Project project : super.projects) {
			if (project.getStart().getMonth() + 1 == 11) {
				reward += rewardForEachProject;
			}
		}
		super.rewardPrice = reward;
	}
	// -----------------------------------------------

}
