package org.bihe.beans;

import java.util.Date;
import java.util.concurrent.TimeUnit;

public class Factor {
	private Customer customer;
	private Movie movie;
	private Date borrowDate;
	private int dueDays = 4;
	private Date returnDate;
	private float basePrice = 500;
	private int factorNo;
	private static int basedNo = 1;
	private boolean returnStatus;
	private float pay;

	// --------------------------------------
	// constructor
	public Factor(Customer customer, Movie movie, Date borrowDate, Date returnDate) {
		this.customer = customer;
		this.movie = movie;
		this.borrowDate = borrowDate;
		this.returnDate = returnDate;
		this.factorNo = basedNo;
		basedNo++;
	}

	// ---------------------------------------
	// methods
	// is the customer return the movie
	public void isReturned() {
		if (returnDate != null) {
			this.returnStatus = true;
		}
	}

	// ------------------
	// the movie borrowed at least once.
	public void hasBorrowed() {
		boolean status = true;
		this.movie.setHasEverBeenBorrowed(status);
	}

	// -----------------------
	// the customer is punctual or unfaithfull?
	public void punctualStatus() {
		int n = 0;
		if (returnDate == null) {
			n = (int) TimeUnit.DAYS.convert(new Date().getTime() - borrowDate.getTime(), TimeUnit.MILLISECONDS);

		} else {
			n = (int) TimeUnit.DAYS.convert(returnDate.getTime() - borrowDate.getTime(), TimeUnit.MILLISECONDS);
		}
		if (n > dueDays) {
			customer.setPunctual(false);
		}
	}
	//---------------------------------
	//the coustumer return movie and want to know how much he should payed for each film.
	public void howMuchDoesItCosts(){
		pay = basePrice;
		float extra=0;
		int n =0;
		if (returnDate == null) {
			n = (int) TimeUnit.DAYS.convert(new Date().getTime() - borrowDate.getTime(), TimeUnit.MILLISECONDS);

		} else {
			n = (int) TimeUnit.DAYS.convert(returnDate.getTime() - borrowDate.getTime(), TimeUnit.MILLISECONDS);
		}
		if (n > dueDays) {
			extra = movie.getPrice()*(n-dueDays);
		}
		pay = pay+extra;
	}
	// ----------------------------------------
	// Accessors
	public Customer getCustomer() {
		return this.customer;
	}

	public Movie getMovie() {
		return this.movie;
	}

	public Date getBorrowDate() {
		return this.borrowDate;
	}

	public Date getReturnDate() {
		return this.returnDate;
	}

	public float getBasePrice() {
		return this.basePrice;
	}

	public int getFactorNo() {
		return this.factorNo;
	}

	public boolean getReturnStatus() {
		return this.returnStatus;
	}

	public int getDueDays() {
		return this.dueDays;
	}
	public float getPay(){
		return this.pay;
	}

	// ------------------------------------------
	// Mutators
	public void setCustomer(Customer customer) {
		this.customer = customer;
	}

	public void setMovie(Movie movie) {
		this.movie = movie;
	}

	public void setBorrowDate(Date borrowDate) {
		this.borrowDate = borrowDate;
	}

	public void setReturnDate(Date returnDate) {
		this.returnDate = returnDate;
	}

	public void setBasePrice(float basePrice) {
		this.basePrice = basePrice;
	}

	public void setDueDays(int dueDays) {
		this.dueDays = dueDays;
	}
	public void setPay(float pay){
		this.pay = pay;
	}
	// ---------------------------------------

	// override
	//equals method
	@Override
	public boolean equals(Object o) {
		if (o.equals(null)) {
			return false;
		}
		if (o instanceof Factor) {
			Factor f = (Factor) o;
			if (this.borrowDate.equals(f.getBorrowDate()) && this.returnDate.equals(f.getReturnDate())
					&& this.basePrice == f.getBasePrice() && this.customer.equals(f.getMovie())
					&& this.movie.equals(f.getMovie()) && this.factorNo == f.getFactorNo()) {
				return true;
			}
		}
		return false;
	}
	//------------------------------------
	//toString method
	public String toString(){
		return "Customer: "+this.customer.toString()+" - Movie: "+this.movie.toString();
	}

}
