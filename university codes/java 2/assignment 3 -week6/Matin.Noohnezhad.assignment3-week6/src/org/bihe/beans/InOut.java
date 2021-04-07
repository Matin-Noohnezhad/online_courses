package org.bihe.beans;

import java.util.Date;

public class InOut {
	private Date in;
	private Date out;
	private Employee employee;
	private int sumOfHours;

	// --------------------------------
	// constructor
	public InOut(Date in, Date out, Employee employee) {
		this.in = in;
		this.out = out;
		this.employee = employee;
		int y = out.getYear() - in.getYear();
		int n = out.getMonth() - in.getMonth();
		int m = out.getDay() - in.getDay();
		int z = out.getHours() - in.getHours();
		this.sumOfHours = (y * 24 * 365) + (n * 24 * 30) + (m * 24) + z;

	}

	// --------------------------------
	// Accessors
	public Date getIn() {
		return this.in;
	}

	public Date getOut() {
		return this.out;
	}

	public Employee getEmployee() {
		return this.employee;
	}

	public int getSumOfHours() {
		return this.sumOfHours;
	}

	// --------------------------------
	// Mutators
	public void setIn(Date in) {
		this.in = in;
	}

	public void setOut(Date out) {
		this.out = out;
	}

	public void setEmployee(Employee employee) {
		this.employee = employee;
	}

	public void setSumOfHours(int sumOfHours) {
		this.sumOfHours = sumOfHours;
	}
}
