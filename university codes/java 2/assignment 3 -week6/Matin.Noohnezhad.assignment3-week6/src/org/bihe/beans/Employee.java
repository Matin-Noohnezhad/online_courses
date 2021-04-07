package org.bihe.beans;

import java.lang.invoke.ConstantCallSite;
import java.util.ArrayList;

public abstract class Employee implements CalculateSalary {
	protected String firstName;
	protected String lastName;
	protected int hoursInDay;
	protected int daysInMonth;
	protected float incomeInHour;
	protected int insurancePercent;
	protected int taxPercent;
	protected ArrayList<InOut> workHours = new ArrayList<>();
	protected float baseSalary;
	protected float insurancePay;
	protected float extraHoursMultipledBy;
	protected float extraHoursPay;
	protected ArrayList<Project> projects = new ArrayList<>();
	protected ArrayList<SellDetail> sales = new ArrayList<>();
	protected float rewardPrice;
	protected float impureIncome;
	protected float floorOfMoneyForTax;
	protected float taxPay;
	protected float pureIncome;

	// -------------------------------------------------------------------
	// constructor
	public Employee(String firstName, String lastName) {
		this.firstName = firstName;
		this.lastName = lastName;
		this.hoursInDay = 8;
		this.daysInMonth = 21;
	}

	// ------------------------------------------------------------------
	// method
	public void eachInOutPutInArray(InOut inout) {
		workHours.add(inout);
	}

	public void eachProjectPutInArray(Project project) {
		projects.add(project);
	}

	public void eachSellingPutInArray(SellDetail sellDetail) {
		sales.add(sellDetail);
	}

	// ---------------------------------------------------------------------
	// Accessors
	public String getFirstName() {
		return firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public int getHoursInDay() {
		return hoursInDay;
	}

	public int getDaysInMonth() {
		return daysInMonth;
	}

	public float getIncomeInHour() {
		return incomeInHour;
	}

	public int getInsurancePercent() {
		return insurancePercent;
	}

	public int getTaxPercent() {
		return taxPercent;
	}

	public ArrayList<InOut> getWorkHours() {
		return workHours;
	}

	public float getBaseSalary() {
		return baseSalary;
	}

	public float getInsurancePay() {
		return insurancePay;
	}

	public float getExtraHoursMultipledBy() {
		return extraHoursMultipledBy;
	}

	public float getExtraHoursPay() {
		return extraHoursPay;
	}

	public ArrayList<Project> getProjects() {
		return projects;
	}

	public ArrayList<SellDetail> getSales() {
		return sales;
	}

	public float getRewardPrice() {
		return rewardPrice;
	}

	public float getImpureIncome() {
		return impureIncome;
	}

	public float getFloorOfMoneyForTax() {
		return floorOfMoneyForTax;
	}

	public float getTaxPay() {
		return taxPay;
	}

	public float getPureIncome() {
		return pureIncome;
	}

	// ------------------------------------------------------------------------
	// Mutators
	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public void setHoursInDay(int hoursInDay) {
		this.hoursInDay = hoursInDay;
	}

	public void setDaysInMonth(int daysInMonth) {
		this.daysInMonth = daysInMonth;
	}

	public void setIncomeInHour(float incomeInHour) {
		this.incomeInHour = incomeInHour;
	}

	public void setInsurancePercent(int insurancePercent) {
		this.insurancePercent = insurancePercent;
	}

	public void setTaxPercent(int taxPercent) {
		this.taxPercent = taxPercent;
	}

	public void setWorkHours(ArrayList<InOut> workHours) {
		this.workHours = workHours;
	}

	public void setBaseSalary(float baseSalary) {
		this.baseSalary = baseSalary;
	}

	public void setInsurancePay(float insurancePay) {
		this.insurancePay = insurancePay;
	}

	public void setExtraHoursMultipledBy(float extraHoursMultipledBy) {
		this.extraHoursMultipledBy = extraHoursMultipledBy;
	}

	public void setExtraHoursPay(float extraHoursPay) {
		this.extraHoursPay = extraHoursPay;
	}

	public void setProjects(ArrayList<Project> projects) {
		this.projects = projects;
	}

	public void setSales(ArrayList<SellDetail> sales) {
		this.sales = sales;
	}

	public void setRewardPrice(float awardPrice) {
		this.rewardPrice = awardPrice;
	}

	public void setImpureIncome(float impureIncome) {
		this.impureIncome = impureIncome;
	}

	public void setFloorOfMoneyForTax(float floorOfMoneyForTax) {
		this.floorOfMoneyForTax = floorOfMoneyForTax;
	}

	public void setTaxPay(float taxPay) {
		this.taxPay = taxPay;
	}

	public void setPureIncome(float pureIncome) {
		this.pureIncome = pureIncome;
	}

	// -----------------------------------------------------------------------------
	// override interface methods
	@Override
	public void calcBaseSalary() {
		int n = this.hoursInDay * this.daysInMonth;
		this.baseSalary = this.incomeInHour * n;
	}

	@Override
	public void calcTax() {
		float f = this.impureIncome - this.insurancePay;
		if (f > this.floorOfMoneyForTax) {
			this.taxPay = f * this.taxPercent / 100;
		}
	}

	@Override
	public abstract void calcReward();

	@Override
	public void calcExtraHoursPay() {
		int sum = 0;
		for (InOut inOut : this.workHours) {
			if (inOut.getIn().getMonth() + 1 == 11) {
				sum += inOut.getSumOfHours();
			}
		}
		if (sum > this.hoursInDay * this.daysInMonth) {
			this.extraHoursPay = (sum - this.hoursInDay * this.daysInMonth) * (this.extraHoursMultipledBy - 1)
					* this.incomeInHour;
		}

	}

	@Override
	public void calcInsurance() {
		this.insurancePay = this.baseSalary * this.insurancePercent / 100;
	}

	@Override
	public void calcImpureIncome() {
		int sum = 0;
		for (InOut inOut : this.workHours) {
			if (inOut.getIn().getMonth() + 1 == 11) {
				sum += inOut.getSumOfHours();
			}
		}
		float f = (sum * this.incomeInHour) + this.extraHoursPay + this.rewardPrice;
		this.impureIncome = f;

	}

	public void calcPureIncome() {
		this.pureIncome = this.impureIncome - this.insurancePay - this.taxPay;
	}

	// -----------------------------------------------------------------------------------
	// override toString and equals method
	// toString method
	public String toString() {
		return this.firstName + " " + this.lastName;
	}

	// ------------------------------------------
	// equals method
	public boolean equals(Object o) {
		if (o.equals(null)) {
			return false;
		}
		if (o instanceof Employee) {
			Employee e = (Employee) o;
			if (this.firstName.equals(e.getFirstName()) && this.lastName.equals(e.getLastName())) {
				return true;
			}
		}
		return false;
	}
	// ------------------------------------------------------------------------------------
}
