package org.bihe.service;

import java.util.ArrayList;

import org.bihe.beans.Customer;
import org.bihe.beans.Data;
import org.bihe.beans.Factor;
import org.bihe.beans.Movie;

public class Service {
	private static ArrayList<Customer> customers = new ArrayList<>();
	private static ArrayList<Movie> movies = new ArrayList<>();
	private static ArrayList<Factor> invoices = new ArrayList<>();

	// --------------------------
	// method
	public static void added() {
		Data data = new Data();
		data.data();
		movies.addAll(data.getMovies());
		invoices.addAll(data.getInvoices());
		customers.addAll(data.getCustomers());

	}

	// -----------------------------
	// Accessors
	public ArrayList<Customer> getCustomers() {
		return this.customers;
	}

	public ArrayList<Movie> getMovies() {
		return this.movies;
	}

	public ArrayList<Factor> getInvoices() {
		return this.invoices;
	}

	// ----------------------------------
	// Mutators
	public void setCustomers(ArrayList<Customer> customers) {
		this.customers = customers;
	}

	public void setMovies(ArrayList<Movie> movies) {
		this.movies = movies;
	}

	public void setInvoices(ArrayList<Factor> invoices) {
		this.invoices = invoices;
	}

	// --------------------------------------
	public void Question1() {
		if (customers.size() == 0) {
			added();
		}
		System.out.println("--------------------------------------");
		System.out.println("The answer of question1 is :");
		System.out.print("The customers are: ");
		for (Customer customer : customers) {
			System.out.print(customer.toString() + " - ");
		}
		System.out.println();
		System.out.println("--------------------------------------");
	}
	// -----------------------------

	public void Question2() {
		if (customers.size() == 0) {
			added();
		}
		System.out.println("--------------------------------------");
		System.out.println("The answer of question2 is :");
		System.out.println("The list of person which borrowed film and has not returned yet.");
		ArrayList<Factor> factors = new ArrayList<>();
		for (Factor invoice : invoices) {
			invoice.isReturned();
			if (invoice.getReturnStatus() == false) {
				
				System.out.print(invoice.getCustomer().toString() + " - ");
			}

		}
		System.out.println();
		System.out.println("--------------------------------------");
	}

	public void Question3() {
		if (customers.size() == 0) {
			added();
		}
		System.out.println("--------------------------------------");
		System.out.println("The answer of question3 is :");
		System.out.println("The list of films that never been borrowed: ");
		for (Factor invoice : invoices) {
			invoice.hasBorrowed();
		}
		for (Movie movie : movies) {
			if (movie.getHasEverBeenBorrowed() == false) {
				System.out.print(movie.toString() + " - ");
			}
		}
		System.out.println();
		System.out.println("--------------------------------------");
	}

	 public void Question4() {
	 if(customers.size()==0){
	 added();
	 }
	 System.out.println("--------------------------------------");
	 System.out.println("The answer of question4 is :");
	 System.out.println("The list of punctual customers: ");
	 for(Factor invoice: invoices){
		 invoice.punctualStatus();
	 }
	 for(Customer customer: customers){
		 if(customer.IsPunctual()){
			 System.out.print(customer.toString()+" - ");
		 }
	 }
	 System.out.println();
	 System.out.println("--------------------------------------");
	 }
	
	 public void Question5() {
	 if(customers.size()==0){
	 added();
	 }
	 System.out.println("--------------------------------------");
	 System.out.println("The answer of question5 is :");
	 for(Customer customer:customers){
		 System.out.println("The films that "+customer.toString()+" borrowed is:");
		 for(Factor invoice: invoices){
			 invoice.isReturned();
		 }
		 for(int i =0;i<customer.getInvoices().size();i++){
			 System.out.print(customer.getInvoices().get(i).getMovie().toString());
			if(customer.getInvoices().get(i).getReturnStatus()==false){
				System.out.print("(not return) - ");
			}else{
				System.out.print("(return) - ");
			}
		 }
		 System.out.println();
	 }
	 System.out.println();
	 System.out.println("--------------------------------------");
	 }
	
	 public void Question6() {
	 if(customers.size()==0){
	 added();
	 }
	 System.out.println("--------------------------------------");
	 System.out.println("The answer of question6 is :");
	 System.out.println("if any of this person want to pay the money for all the film that he returned before or return now, he must paid: ");
	 for(Factor invoice:invoices){
		 invoice.howMuchDoesItCosts();
	 }
	 for(Customer customer:customers){
		 float moneyPayForAllFactor=0;
		 System.out.print("the money that "+customer.toString()+" should pay is: ");
		 for(Factor invoice:customer.getInvoices()){
			 moneyPayForAllFactor += invoice.getPay();
		 }
		 System.out.println(moneyPayForAllFactor);
		 
	 }
	 System.out.println("--------------------------------------");
	 }

}
