package org.bihe.beans;

import java.util.ArrayList;

public class Customer {
	private String firstName;
	private String lastName;
	private String phoneNo;
	private int ID;
	private ArrayList<Factor> invoices = new ArrayList<>();
	private static int basedNo = 1;
	private boolean punctual = true;

	// -------------------------------------
	// constructor
	public Customer(String firstName, String lastName, String phoneNo) {
		this.firstName = firstName;
		this.lastName = lastName;
		this.phoneNo = phoneNo;
		this.ID = basedNo;
		basedNo++;
	}

	// -------------------------------------
	// methods
	// each invoice that built was put in it's customer arrayList of factor
	public void setEachInvoice(Factor invoice) {
		this.invoices.add(invoice);
	}
	//---------------------------------------
	// Accessors
	public String getFirstName() {
		return this.firstName;
	}

	public String getLastName() {
		return this.lastName;
	}

	public String getPhoneNo() {
		return this.phoneNo;
	}

	public int getID() {
		return this.ID;
	}

	public ArrayList<Factor> getInvoices() {
		return this.invoices;
	}

	public static int getBasedNo() {
		return basedNo;
	}

	public boolean IsPunctual() {
		return this.punctual;
	}

	// --------------------------------------
	// Mutators
	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public void setPhoneNo(String phoneNo) {
		this.phoneNo = phoneNo;
	}

	public void setInvoices(ArrayList<Factor> invoices) {
		this.invoices = invoices;
	}

	public void setPunctual(boolean status) {
		this.punctual = status;
	}

	// ---------------------------------------
	// override
	//equals method
	@Override
	public boolean equals(Object o) {
		if (o.equals(null)) {
			return false;
		}
		if (o instanceof Customer) {
			Customer c = (Customer) o;
			if (this.firstName.equals(c.getFirstName()) && this.lastName.equals(c.getLastName())
					&& this.phoneNo.equals(c.getPhoneNo()) && this.invoices.equals(c.getInvoices())
					&& this.ID == c.getID()) {
				return true;
			}
		}
		return false;
	}
	//-------------------------------------------
	//toString method
	public String toString(){
		return this.firstName+" "+this.lastName;
	}
	
}
