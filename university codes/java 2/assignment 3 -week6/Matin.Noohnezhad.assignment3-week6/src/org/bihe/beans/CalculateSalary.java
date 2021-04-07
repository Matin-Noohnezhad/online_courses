package org.bihe.beans;

public interface CalculateSalary {
	// fixed
	public void calcBaseSalary();
	//fixed
	public void calcTax();

	// fixed - should be override for each child class(the commercial manager
	// and sale manager don't needed.just put it empty in that classes)
	public void calcReward();

	// fixed
	public void calcExtraHoursPay();

	// fixed
	public void calcInsurance();

	// fixed
	public void calcImpureIncome();
	//fixed
	public void calcPureIncome();
}
