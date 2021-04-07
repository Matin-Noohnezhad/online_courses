package org.bihe.beans;

import java.util.ArrayList;
import java.util.Date;

public class Data {
	ArrayList<Customer> customers = new ArrayList<>();
	ArrayList<Movie> movies = new ArrayList<>();
	ArrayList<Factor> invoices = new ArrayList<>();

	// -------------------------------
	

	public void data() {
		// customer
		Customer c1 = new Customer("Matin", "Noohnezhad", "0917362");
		customers.add(c1);
		Customer c2 = new Customer("Farzam", "Sabeti", "0938472");
		customers.add(c2);
		Customer c3 = new Customer("Faran", "Sanaei", "0943765");
		customers.add(c3);
		Customer c4 = new Customer("Erfan", "Sameiyian", "0998723");
		customers.add(c4);
		Customer c5 = new Customer("Mona", "Momtazian", "0918971");
		customers.add(c5);
		Customer c6 = new Customer("Soroosh", "Vedaei", "0913425");
		customers.add(c6);
		// -------------------------------------------------
		// movie
		Movie m1 = new Movie("PulpFiction", Genre.thriller, 7, 1000.0f);
		movies.add(m1);
		Movie m2 = new Movie("CitizenKane", Genre.drama, 5, 1000.0f);
		movies.add(m2);
		Movie m3 = new Movie("ClockWorkOrange", Genre.scienceFiction, 6, 1000.0f);
		movies.add(m3);
		Movie m4 = new Movie("SilverLiningsPlaybook", Genre.romance, 5, 1000.0f);
		movies.add(m4);
		Movie m5 = new Movie("SavingPrivateRyan", Genre.war, 5, 1000.0f);
		movies.add(m5);
		Movie m6 = new Movie("TheShining", Genre.horror, 5, 1000.0f);
		movies.add(m6);
		Movie m7 = new Movie("TheGodfather", Genre.drama, 3, 1000.0f);
		movies.add(m7);
		Movie m8 = new Movie("TheDarkKnight", Genre.thriller, 4, 1000.0f);
		movies.add(m8);
		Movie m9 = new Movie("TheShawshankRedemption", Genre.drama, 2, 1000.0f);
		movies.add(m9);
		// -------------------------------------------------
		// Factor
		Factor f1 = new Factor(customers.get(0), movies.get(0), new Date("11/1/2016"), new Date("11/4/2016"));
		invoices.add(f1);
		Factor f2 = new Factor(customers.get(0), movies.get(1), new Date("11/1/2016"), new Date("11/3/2016"));
		invoices.add(f2);
		Factor f3 = new Factor(customers.get(0), movies.get(2), new Date("11/1/2016"), new Date("11/5/2016"));
		invoices.add(f3);
		Factor f4 = new Factor(customers.get(1), movies.get(1), new Date("11/2/2016"), new Date("11/6/2016"));
		invoices.add(f4);
		Factor f5 = new Factor(customers.get(1), movies.get(2), new Date("11/2/2016"), null);
		invoices.add(f5);
		Factor f6 = new Factor(customers.get(1), movies.get(3), new Date("11/2/2016"), new Date("11/5/2016"));
		invoices.add(f6);
		Factor f7 = new Factor(customers.get(2), movies.get(2), new Date("11/4/2016"), new Date("11/8/2016"));
		invoices.add(f7);
		Factor f8 = new Factor(customers.get(2), movies.get(3), new Date("11/3/2016"), new Date("11/7/2016"));
		invoices.add(f8);
		Factor f9 = new Factor(customers.get(2), movies.get(4), new Date("11/2/2016"), new Date("11/6/2016"));
		invoices.add(f9);
		Factor f10 = new Factor(customers.get(3), movies.get(3), new Date("11/3/2016"), new Date("11/6/2016"));
		invoices.add(f10);
		Factor f11 = new Factor(customers.get(3), movies.get(4), new Date("11/1/2016"), new Date("11/5/2016"));
		invoices.add(f11);
		Factor f12 = new Factor(customers.get(3), movies.get(5), new Date("11/3/2016"), null);
		invoices.add(f12);
		Factor f13 = new Factor(customers.get(4), movies.get(4), new Date("11/2/2016"), null);
		invoices.add(f13);
		Factor f14 = new Factor(customers.get(4), movies.get(5), new Date("11/4/2016"), new Date("11/6/2016"));
		invoices.add(f14);
		Factor f15 = new Factor(customers.get(4), movies.get(0), new Date("11/5/2016"), new Date("11/7/2016"));
		invoices.add(f15);
		Factor f16 = new Factor(customers.get(5), movies.get(5), new Date("11/1/2016"), new Date("11/7/2016"));
		invoices.add(f16);
		Factor f17 = new Factor(customers.get(5), movies.get(0), new Date("11/3/2016"), new Date("11/7/2016"));
		invoices.add(f17);
		Factor f18 = new Factor(customers.get(5), movies.get(1), new Date("11/2/2016"), new Date("11/7/2016"));
		invoices.add(f18);
		// ---------------------------------
		forEachInvoice(invoices);
		// ---------------------------------

	}
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
		//--------------------------------
		//Mutators
		public void setCustomers(ArrayList<Customer> customers){
			this.customers = customers;
		}
		public void setMovies(ArrayList<Movie> movies){
			this.movies = movies;
		}
		public void setInvoices(ArrayList<Factor> invoices){
			this.invoices = invoices;
		}
		//------------------------------------

	// ------------------------------------------------------
	// set each factor to customer's arrayList of factor
	public static void forEachInvoice(ArrayList<Factor> invoices) {
		for (Factor f : invoices) {
			f.getCustomer().setEachInvoice(f);
		}
	}
}
