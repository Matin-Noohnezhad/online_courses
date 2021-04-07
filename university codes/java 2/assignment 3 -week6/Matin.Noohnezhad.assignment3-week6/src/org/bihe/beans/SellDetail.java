package org.bihe.beans;

import java.util.Date;

public class SellDetail {
	private Date sellDate;
	private Project project;
	private CommercialEmployee commercialEmployee;
	private SaleManager saleManager;

	// ----------------------------------------------
	// constructor
	public SellDetail(Date sellDate, Project project, CommercialEmployee commercialEmployee) {
		this.sellDate = sellDate;
		this.project = project;
		this.commercialEmployee = commercialEmployee;
		this.saleManager = project.getSaleManager();
	}

	// -----------------------------------------------
	// Accessors
	public Date getSellDate() {
		return sellDate;
	}

	public Project getProject() {
		return project;
	}

	public CommercialEmployee getCommercialEmployee() {
		return commercialEmployee;
	}

	// -------------------------------------------------
	// Mutators
	public void setSellDate(Date sellDate) {
		this.sellDate = sellDate;
	}

	public void setProject(Project project) {
		this.project = project;
	}

	public void setCommercialEmployee(CommercialEmployee commercialEmployee) {
		this.commercialEmployee = commercialEmployee;
	}

	// -------------------------------------------------------------------------
	// override toString method
	public String toString() {

		return ("Project name: " + project.getName() + " - sell's date: " + this.sellDate
				+ " - sale manager of this project is: " + project.getSaleManager().toString()
				+ " - commercial employee which sell the project is: " + this.commercialEmployee.toString());
	}
	// -------------------------------------------------------------------------
}