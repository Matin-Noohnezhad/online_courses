package org.bihe.beans;

import java.util.ArrayList;
import java.util.Date;

public class Data {
	ArrayList<Employee> allEmployees = new ArrayList<>();
	ArrayList<ProductManager> productManagers = new ArrayList<>();
	ArrayList<TechnicalManager> technicalManagers = new ArrayList<>();
	ArrayList<ProductEmployee> productEmployees = new ArrayList<>();
	ArrayList<CommercialManager> commercialManagers = new ArrayList<>();
	ArrayList<SaleManager> saleManagers = new ArrayList<>();
	ArrayList<CommercialEmployee> commercialEmployees = new ArrayList<>();
	ArrayList<InOut> inOuts = new ArrayList<>();
	ArrayList<Project> projects = new ArrayList<>();
	ArrayList<SellDetail> sellDetails = new ArrayList<>();

	// ----------------------------------------------------------------------
	public void data() {
		ProductManager pm1 = new ProductManager("Cristiano", "Ronaldo");
		productManagers.add(pm1);
		ProductManager pm2 = new ProductManager("Lionel", "Messi");
		productManagers.add(pm2);
		TechnicalManager tm1 = new TechnicalManager("Jose", "Mourinho");
		technicalManagers.add(tm1);
		TechnicalManager tm2 = new TechnicalManager("Pep", "Guardiola");
		technicalManagers.add(tm2);
		ProductEmployee pe1 = new ProductEmployee("Lebron", "James");
		productEmployees.add(pe1);
		ProductEmployee pe2 = new ProductEmployee("Kobe", "Bryant");
		productEmployees.add(pe2);
		ProductEmployee pe3 = new ProductEmployee("Wilt", "Chamberlain");
		productEmployees.add(pe3);
		ProductEmployee pe4 = new ProductEmployee("Stephen", "Curry");
		productEmployees.add(pe4);
		CommercialManager cm1 = new CommercialManager("Larry", "Bird");
		commercialManagers.add(cm1);
		CommercialManager cm2 = new CommercialManager("Magic", "Johnson");
		commercialManagers.add(cm2);
		SaleManager sm1 = new SaleManager("Dirk", "Nowitzki");
		saleManagers.add(sm1);
		SaleManager sm2 = new SaleManager("Miroslav", "Klose");
		saleManagers.add(sm2);
		SaleManager sm3 = new SaleManager("Michael", "Jordan");
		saleManagers.add(sm3);
		SaleManager sm4 = new SaleManager("Karl", "Malone");
		saleManagers.add(sm4);
		CommercialEmployee ce1 = new CommercialEmployee("Earvin", "Ngapeth");
		commercialEmployees.add(ce1);
		CommercialEmployee ce2 = new CommercialEmployee("Ivan", "Zaytsev");
		commercialEmployees.add(ce2);
		CommercialEmployee ce3 = new CommercialEmployee("Matthew", "Anderson");
		commercialEmployees.add(ce3);
		CommercialEmployee ce4 = new CommercialEmployee("Mariusz", "Wlazly");
		commercialEmployees.add(ce4);
		// ------------------------------
		// add to allEmployees
		allEmployees.addAll(productManagers);
		allEmployees.addAll(technicalManagers);
		allEmployees.addAll(productEmployees);
		allEmployees.addAll(commercialManagers);
		allEmployees.addAll(saleManagers);
		allEmployees.addAll(commercialEmployees);
		// ------------------------------
		// inouts
		// firstperson
		InOut io1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io1);
		InOut io2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io2);
		InOut io3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(0));
		inOuts.add(io3);
		InOut io4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io4);
		InOut io5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(0));
		inOuts.add(io5);
		InOut io6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(0));
		inOuts.add(io6);
		InOut io7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(0));
		inOuts.add(io7);
		InOut io8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io8);
		InOut io9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io9);
		InOut io10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io10);
		InOut io11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(0));
		inOuts.add(io11);
		InOut io12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io12);
		InOut io13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(0));
		inOuts.add(io13);
		InOut io14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(0));
		inOuts.add(io14);
		InOut io15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io15);
		InOut io16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(0));
		inOuts.add(io16);
		InOut io17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(0));
		inOuts.add(io17);
		InOut io18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(0));
		inOuts.add(io18);
		InOut io19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(0));
		inOuts.add(io19);
		InOut io20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io20);
		InOut io21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io21);
		InOut io22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(0));
		inOuts.add(io22);
		InOut io23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(0));
		inOuts.add(io23);
		// second person
		InOut i2o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o1);
		InOut i2o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o2);
		InOut i2o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(1));
		inOuts.add(i2o3);
		InOut i2o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o4);
		InOut i2o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(1));
		inOuts.add(i2o5);
		InOut i2o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(1));
		inOuts.add(i2o6);
		InOut i2o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o8);
		InOut i2o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o9);
		InOut i2o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o10);
		InOut i2o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(1));
		inOuts.add(i2o11);
		InOut i2o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(1));
		inOuts.add(i2o13);
		InOut i2o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(1));
		inOuts.add(i2o14);
		InOut i2o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o15);
		InOut i2o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(1));
		inOuts.add(i2o16);
		InOut i2o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(1));
		inOuts.add(i2o17);
		InOut i2o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(1));
		inOuts.add(i2o18);
		InOut i2o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(1));
		inOuts.add(i2o19);
		InOut i2o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o20);
		InOut i2o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o21);
		InOut i2o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(1));
		inOuts.add(i2o22);
		InOut i2o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(1));
		inOuts.add(i2o23);
		// third person
		InOut i3o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o1);
		InOut i3o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(2));
		inOuts.add(i3o3);
		InOut i3o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o4);
		InOut i3o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(2));
		inOuts.add(i3o5);
		InOut i3o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o8);
		InOut i3o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o9);
		InOut i3o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o10);
		InOut i3o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(2));
		inOuts.add(i3o11);
		InOut i3o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o12);
		InOut i3o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(2));
		inOuts.add(i3o14);
		InOut i3o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o15);
		InOut i3o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(2));
		inOuts.add(i3o16);
		InOut i3o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(2));
		inOuts.add(i3o17);
		InOut i3o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(2));
		inOuts.add(i3o18);
		InOut i3o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(2));
		inOuts.add(i3o19);
		InOut i3o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o20);
		InOut i3o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o21);
		InOut i3o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(2));
		inOuts.add(i3o22);
		InOut i3o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(2));
		inOuts.add(i3o23);
		// fourth person
		InOut i4o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o1);
		InOut i4o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o2);
		InOut i4o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(3));
		inOuts.add(i4o3);
		InOut i4o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o4);
		InOut i4o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(3));
		inOuts.add(i4o5);
		InOut i4o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(3));
		inOuts.add(i4o6);
		InOut i4o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(3));
		inOuts.add(i4o7);
		InOut i4o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o8);
		InOut i4o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o9);
		InOut i4o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o10);
		InOut i4o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(3));
		inOuts.add(i4o11);
		InOut i4o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o12);
		InOut i4o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(3));
		inOuts.add(i4o14);
		InOut i4o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o15);
		InOut i4o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(3));
		inOuts.add(i4o16);
		InOut i4o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(3));
		inOuts.add(i4o17);
		InOut i4o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(3));
		inOuts.add(i4o18);
		InOut i4o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(3));
		inOuts.add(i4o19);
		InOut i4o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o20);
		InOut i4o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o21);
		InOut i4o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(3));
		inOuts.add(i4o22);
		InOut i4o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(3));
		inOuts.add(i4o23);
		// fifth person
		InOut i5o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o1);
		InOut i5o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o2);
		InOut i5o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(4));
		inOuts.add(i5o3);
		InOut i5o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o4);
		InOut i5o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(4));
		inOuts.add(i5o5);
		InOut i5o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(4));
		inOuts.add(i5o6);
		InOut i5o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(4));
		inOuts.add(i5o7);
		InOut i5o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o8);
		InOut i5o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o9);
		InOut i5o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o10);
		InOut i5o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(4));
		inOuts.add(i5o11);
		InOut i5o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o12);
		InOut i5o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(4));
		inOuts.add(i5o13);
		InOut i5o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(4));
		inOuts.add(i5o14);
		InOut i5o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o15);
		InOut i5o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(4));
		inOuts.add(i5o16);
		InOut i5o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(4));
		inOuts.add(i5o17);
		InOut i5o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(4));
		inOuts.add(i5o18);
		InOut i5o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(4));
		inOuts.add(i5o19);
		InOut i5o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o20);
		InOut i5o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o21);
		InOut i5o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(4));
		inOuts.add(i5o22);
		InOut i5o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(4));
		inOuts.add(i5o23);
		// sixth person
		InOut i6o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o1);
		InOut i6o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o2);
		InOut i6o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(5));
		inOuts.add(i6o3);
		InOut i6o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o4);
		InOut i6o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(5));
		inOuts.add(i6o5);
		InOut i6o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(5));
		inOuts.add(i6o6);
		InOut i6o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(5));
		inOuts.add(i6o7);
		InOut i6o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o8);
		InOut i6o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o9);
		InOut i6o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o10);
		InOut i6o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(5));
		inOuts.add(i6o11);
		InOut i6o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o12);
		InOut i6o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(5));
		inOuts.add(i6o13);
		InOut i6o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(5));
		inOuts.add(i6o14);
		InOut i6o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o15);
		InOut i6o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(5));
		inOuts.add(i6o16);
		InOut i6o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(5));
		inOuts.add(i6o17);
		InOut i6o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(5));
		inOuts.add(i6o18);
		InOut i6o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(5));
		inOuts.add(i6o19);
		InOut i6o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o20);
		InOut i6o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o21);
		InOut i6o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(5));
		inOuts.add(i6o22);
		InOut i6o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(5));
		inOuts.add(i6o23);
		// seventh person
		InOut i7o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o2);
		InOut i7o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(6));
		inOuts.add(i7o3);
		InOut i7o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o4);
		InOut i7o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(6));
		inOuts.add(i7o5);
		InOut i7o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(6));
		inOuts.add(i7o6);
		InOut i7o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(6));
		inOuts.add(i7o7);
		InOut i7o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o8);
		InOut i7o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o10);
		InOut i7o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(6));
		inOuts.add(i7o11);
		InOut i7o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o12);
		InOut i7o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(6));
		inOuts.add(i7o13);
		InOut i7o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(6));
		inOuts.add(i7o14);
		InOut i7o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o15);
		InOut i7o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(6));
		inOuts.add(i7o16);
		InOut i7o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(6));
		inOuts.add(i7o17);
		InOut i7o18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(6));
		inOuts.add(i7o18);
		InOut i7o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(6));
		inOuts.add(i7o19);
		InOut i7o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o20);
		InOut i7o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o21);
		InOut i7o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(6));
		inOuts.add(i7o22);
		InOut i7o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(6));
		inOuts.add(i7o23);
		// eight person
		InOut i8o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o1);
		InOut i8o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o2);
		InOut i8o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(7));
		inOuts.add(i8o3);
		InOut i8o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o4);
		InOut i8o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(7));
		inOuts.add(i8o5);
		InOut i8o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(7));
		inOuts.add(i8o6);
		InOut i8o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(7));
		inOuts.add(i8o7);
		InOut i8o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o9);
		InOut i8o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o10);
		InOut i8o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(7));
		inOuts.add(i8o11);
		InOut i8o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o12);
		InOut i8o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(7));
		inOuts.add(i8o13);
		InOut i8o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(7));
		inOuts.add(i8o14);
		InOut i8o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o15);
		InOut i8o16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(7));
		inOuts.add(i8o16);
		InOut i8o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(7));
		inOuts.add(i8o17);
		InOut i8o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(7));
		inOuts.add(i8o19);
		InOut i8o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o20);
		InOut i8o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o21);
		InOut i8o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(7));
		inOuts.add(i8o22);
		InOut i8o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(7));
		inOuts.add(i8o23);
		// ninth person
		InOut i9o1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o1);
		InOut i9o2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o2);
		InOut i9o3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(8));
		inOuts.add(i9o3);
		InOut i9o4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o4);
		InOut i9o5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(8));
		inOuts.add(i9o5);
		InOut i9o6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(8));
		inOuts.add(i9o6);
		InOut i9o7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(8));
		inOuts.add(i9o7);
		InOut i9o8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o8);
		InOut i9o9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o9);
		InOut i9o10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o10);
		InOut i9o11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(8));
		inOuts.add(i9o11);
		InOut i9o12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o12);
		InOut i9o13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(8));
		inOuts.add(i9o13);
		InOut i9o14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(8));
		inOuts.add(i9o14);
		InOut i9o15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o15);
		InOut i9o17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(8));
		inOuts.add(i9o17);
		InOut i9o19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(8));
		inOuts.add(i9o19);
		InOut i9o20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o20);
		InOut i9o21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o21);
		InOut i9o22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(8));
		inOuts.add(i9o22);
		InOut i9o23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(8));
		inOuts.add(i9o23);
		// tenth person
		InOut iqo3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(9));
		inOuts.add(iqo3);
		InOut iqo4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo4);
		InOut iqo5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(9));
		inOuts.add(iqo5);
		InOut iqo6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(9));
		inOuts.add(iqo6);
		InOut iqo7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(9));
		inOuts.add(iqo7);
		InOut iqo8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo8);
		InOut iqo9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo9);
		InOut iqo10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo10);
		InOut iqo11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(9));
		inOuts.add(iqo11);
		InOut iqo12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo12);
		InOut iqo14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(9));
		inOuts.add(iqo14);
		InOut iqo15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo15);
		InOut iqo16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(9));
		inOuts.add(iqo16);
		InOut iqo18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(9));
		inOuts.add(iqo18);
		InOut iqo19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(9));
		inOuts.add(iqo19);
		InOut iqo20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo20);
		InOut iqo21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo21);
		InOut iqo22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(9));
		inOuts.add(iqo22);
		InOut iqo23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(9));
		inOuts.add(iqo23);
		// eleventh person
		InOut iwo1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo1);
		InOut iwo2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo2);
		InOut iwo3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(10));
		inOuts.add(iwo3);
		InOut iwo6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(10));
		inOuts.add(iwo6);
		InOut iwo7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(10));
		inOuts.add(iwo7);
		InOut iwo8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo8);
		InOut iwo9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo9);
		InOut iwo10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo10);
		InOut iwo11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(10));
		inOuts.add(iwo11);
		InOut iwo12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo12);
		InOut iwo13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(10));
		inOuts.add(iwo13);
		InOut iwo14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(10));
		inOuts.add(iwo14);
		InOut iwo15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo15);
		InOut iwo16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(10));
		inOuts.add(iwo16);
		InOut iwo17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(10));
		inOuts.add(iwo17);
		InOut iwo18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(10));
		inOuts.add(iwo18);
		InOut iwo19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(10));
		inOuts.add(iwo19);
		InOut iwo20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo20);
		InOut iwo21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo21);
		InOut iwo22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(10));
		inOuts.add(iwo22);
		InOut iwo23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(10));
		inOuts.add(iwo23);
		// twelfth person
		InOut ieo1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo1);
		InOut ieo2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo2);
		InOut ieo3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(11));
		inOuts.add(ieo3);
		InOut ieo4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo4);
		InOut ieo5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(11));
		inOuts.add(ieo5);
		InOut ieo6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(11));
		inOuts.add(ieo6);
		InOut ieo7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(11));
		inOuts.add(ieo7);
		InOut ieo8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo8);
		InOut ieo9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo9);
		InOut ieo10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo10);
		InOut ieo11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(11));
		inOuts.add(ieo11);
		InOut ieo12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo12);
		InOut ieo13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(11));
		inOuts.add(ieo13);
		InOut ieo14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(11));
		inOuts.add(ieo14);
		InOut ieo15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo15);
		InOut ieo16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(11));
		inOuts.add(ieo16);
		InOut ieo17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(11));
		inOuts.add(ieo17);
		InOut ieo18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(11));
		inOuts.add(ieo18);
		InOut ieo19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(11));
		inOuts.add(ieo19);
		InOut ieo20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo20);
		InOut ieo21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(11));
		inOuts.add(ieo21);
		InOut ieo22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(11));
		inOuts.add(ieo22);
		// thirtinth person
		InOut iro1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro1);
		InOut iro2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro2);
		InOut iro3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(12));
		inOuts.add(iro3);
		InOut iro4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro4);
		InOut iro5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(12));
		inOuts.add(iro5);
		InOut iro6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(12));
		inOuts.add(iro6);
		InOut iro7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(12));
		inOuts.add(iro7);
		InOut iro8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro8);
		InOut iro9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro9);
		InOut iro10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro10);
		InOut iro11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(12));
		inOuts.add(iro11);
		InOut iro12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro12);
		InOut iro13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(12));
		inOuts.add(iro13);
		InOut iro14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(12));
		inOuts.add(iro14);
		InOut iro15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro15);
		InOut iro16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(12));
		inOuts.add(iro16);
		InOut iro17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(12));
		inOuts.add(iro17);
		InOut iro18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(12));
		inOuts.add(iro18);
		InOut iro19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(12));
		inOuts.add(iro19);
		InOut iro20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro20);
		InOut iro21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro21);
		InOut iro22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(12));
		inOuts.add(iro22);
		InOut iro23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(12));
		inOuts.add(iro23);
		// fourtinth person
		InOut ito1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito1);
		InOut ito2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito2);
		InOut ito3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(13));
		inOuts.add(ito3);
		InOut ito4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito4);
		InOut ito5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(13));
		inOuts.add(ito5);
		InOut ito6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(13));
		inOuts.add(ito6);
		InOut ito7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(13));
		inOuts.add(ito7);
		InOut ito9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito9);
		InOut ito10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito10);
		InOut ito11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(13));
		inOuts.add(ito11);
		InOut ito14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(13));
		inOuts.add(ito14);
		InOut ito15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito15);
		InOut ito17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(13));
		inOuts.add(ito17);
		InOut ito18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(13));
		inOuts.add(ito18);
		InOut ito19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(13));
		inOuts.add(ito19);
		InOut ito20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito20);
		InOut ito21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito21);
		InOut ito22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(13));
		inOuts.add(ito22);
		InOut ito23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(13));
		inOuts.add(ito23);
		// fiftinth person
		InOut iao3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(14));
		inOuts.add(iao3);
		InOut iao4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao4);
		InOut iao5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(14));
		inOuts.add(iao5);
		InOut iao6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(14));
		inOuts.add(iao6);
		InOut iao7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(14));
		inOuts.add(iao7);
		InOut iao8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao8);
		InOut iao9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao9);
		InOut iao10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao10);
		InOut iao11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(14));
		inOuts.add(iao11);
		InOut iao12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao12);
		InOut iao13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(14));
		inOuts.add(iao13);
		InOut iao14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(14));
		inOuts.add(iao14);
		InOut iao15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao15);
		InOut iao18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(14));
		inOuts.add(iao18);
		InOut iao19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(14));
		inOuts.add(iao19);
		InOut iao20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao20);
		InOut iao21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao21);
		InOut iao23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(14));
		inOuts.add(iao23);
		// sixtinth person
		InOut iso1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso1);
		InOut iso2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso2);
		InOut iso4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso4);
		InOut iso5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(15));
		inOuts.add(iso5);
		InOut iso7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(15));
		inOuts.add(iso7);
		InOut iso8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso8);
		InOut iso9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso9);
		InOut iso11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(15));
		inOuts.add(iso11);
		InOut iso12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso12);
		InOut iso13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(15));
		inOuts.add(iso13);
		InOut iso14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(15));
		inOuts.add(iso14);
		InOut iso15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso15);
		InOut iso16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(15));
		inOuts.add(iso16);
		InOut iso17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(15));
		inOuts.add(iso17);
		InOut iso18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(15));
		inOuts.add(iso18);
		InOut iso20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso20);
		InOut iso21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso21);
		InOut iso22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(15));
		inOuts.add(iso22);
		InOut iso23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(15));
		inOuts.add(iso23);
		// seventinth person
		InOut ido2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido2);
		InOut ido3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(16));
		inOuts.add(ido3);
		InOut ido4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido4);
		InOut ido5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(16));
		inOuts.add(ido5);
		InOut ido6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(16));
		inOuts.add(ido6);
		InOut ido7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(16));
		inOuts.add(ido7);
		InOut ido8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido8);
		InOut ido9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido9);
		InOut ido10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido10);
		InOut ido11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(16));
		inOuts.add(ido11);
		InOut ido12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido12);
		InOut ido13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(16));
		inOuts.add(ido13);
		InOut ido14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(16));
		inOuts.add(ido14);
		InOut ido15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido15);
		InOut ido16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(16));
		inOuts.add(ido16);
		InOut ido17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(16));
		inOuts.add(ido17);
		InOut ido18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(16));
		inOuts.add(ido18);
		InOut ido19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(16));
		inOuts.add(ido19);
		InOut ido20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido20);
		InOut ido21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido21);
		InOut ido22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(16));
		inOuts.add(ido22);
		InOut ido23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(16));
		inOuts.add(ido23);
		// eightinth person
		InOut igo1 = new InOut(new Date("11/1/2016 9:0:0"), new Date("11/1/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo1);
		InOut igo2 = new InOut(new Date("11/2/2016 7:0:0"), new Date("11/2/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo2);
		InOut igo3 = new InOut(new Date("11/3/2016 8:0:0"), new Date("11/3/2016 18:0:0"), allEmployees.get(17));
		inOuts.add(igo3);
		InOut igo4 = new InOut(new Date("11/4/2016 9:0:0"), new Date("11/4/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo4);
		InOut igo5 = new InOut(new Date("11/5/2016 9:0:0"), new Date("11/5/2016 14:0:0"), allEmployees.get(17));
		inOuts.add(igo5);
		InOut igo6 = new InOut(new Date("11/6/2016 8:0:0"), new Date("11/6/2016 15:0:0"), allEmployees.get(17));
		inOuts.add(igo6);
		InOut igo7 = new InOut(new Date("11/7/2016 9:0:0"), new Date("11/7/2016 16:0:0"), allEmployees.get(17));
		inOuts.add(igo7);
		InOut igo8 = new InOut(new Date("11/8/2016 10:0:0"), new Date("11/8/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo8);
		InOut igo9 = new InOut(new Date("11/9/2016 9:0:0"), new Date("11/9/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo9);
		InOut igo10 = new InOut(new Date("11/11/2016 9:0:0"), new Date("11/11/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo10);
		InOut igo11 = new InOut(new Date("11/12/2016 8:0:0"), new Date("11/12/2016 18:0:0"), allEmployees.get(17));
		inOuts.add(igo11);
		InOut igo12 = new InOut(new Date("11/13/2016 7:0:0"), new Date("11/13/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo12);
		InOut igo13 = new InOut(new Date("11/14/2016 9:0:0"), new Date("11/14/2016 15:0:0"), allEmployees.get(17));
		inOuts.add(igo13);
		InOut igo14 = new InOut(new Date("11/15/2016 9:0:0"), new Date("11/15/2016 16:0:0"), allEmployees.get(17));
		inOuts.add(igo14);
		InOut igo15 = new InOut(new Date("11/16/2016 10:0:0"), new Date("11/16/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo15);
		InOut igo16 = new InOut(new Date("11/18/2016 10:0:0"), new Date("11/18/2016 16:0:0"), allEmployees.get(17));
		inOuts.add(igo16);
		InOut igo17 = new InOut(new Date("11/20/2016 11:0:0"), new Date("11/20/2016 18:0:0"), allEmployees.get(17));
		inOuts.add(igo17);
		InOut igo18 = new InOut(new Date("11/22/2016 12:0:0"), new Date("11/22/2016 18:0:0"), allEmployees.get(17));
		inOuts.add(igo18);
		InOut igo19 = new InOut(new Date("11/23/2016 7:0:0"), new Date("11/23/2016 19:0:0"), allEmployees.get(17));
		inOuts.add(igo19);
		InOut igo20 = new InOut(new Date("11/24/2016 7:0:0"), new Date("11/24/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo20);
		InOut igo21 = new InOut(new Date("11/27/2016 9:0:0"), new Date("11/27/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo21);
		InOut igo22 = new InOut(new Date("11/28/2016 8:0:0"), new Date("11/28/2016 19:0:0"), allEmployees.get(17));
		inOuts.add(igo22);
		InOut igo23 = new InOut(new Date("11/29/2016 8:0:0"), new Date("11/29/2016 17:0:0"), allEmployees.get(17));
		inOuts.add(igo23);
		// ------------------------------------------------------
		// Projects
		ArrayList<ProductEmployee> pes1 = new ArrayList<>();
		pes1.add(pe1);
		pes1.add(pe2);
		pes1.add(pe3);
		Project p1 = new Project("real madrid", new Date("11/2/2016"), new Date("11/8/2016"), cm1, pm1, tm2, pes1, sm1,
				20000000.0f);
		projects.add(p1);
		ArrayList<ProductEmployee> pes2 = new ArrayList<>();
		pes2.add(pe2);
		pes2.add(pe3);
		pes2.add(pe4);
		Project p2 = new Project("ladecime", new Date("11/8/2016"), new Date("11/15/2016"), cm2, pm1, tm1, pes2, sm2,
				19000000.0f);
		projects.add(p2);
		ArrayList<ProductEmployee> pes3 = new ArrayList<>();
		pes3.add(pe3);
		pes3.add(pe4);
		pes3.add(pe1);
		Project p3 = new Project("ondecima", new Date("11/15/2016"), new Date("11/21/2016"), cm1, pm2, tm1, pes3, sm3,
				18000000.0f);
		projects.add(p3);
		ArrayList<ProductEmployee> pes4 = new ArrayList<>();
		pes4.add(pe4);
		pes4.add(pe1);
		pes4.add(pe2);
		Project p4 = new Project("dudecima", new Date("11/21/2016"), new Date("11/26/2016"), cm2, pm2, tm2, pes4, sm4,
				18500000.0f);
		projects.add(p4);
		// -------------------------------
		// sellDetails
		SellDetail sale1 = new SellDetail(new Date("11/9/2016"), projects.get(0), ce1);
		sellDetails.add(sale1);
		SellDetail sale2 = new SellDetail(new Date("11/10/2016"), projects.get(0), ce2);
		sellDetails.add(sale2);
		SellDetail sale3 = new SellDetail(new Date("11/11/2016"), projects.get(0), ce3);
		sellDetails.add(sale3);
		SellDetail sale4 = new SellDetail(new Date("11/12/2016"), projects.get(0), ce4);
		sellDetails.add(sale4);
		SellDetail sale5 = new SellDetail(new Date("11/15/2016"), projects.get(1), ce1);
		sellDetails.add(sale5);
		SellDetail sale6 = new SellDetail(new Date("11/16/2016"), projects.get(1), ce2);
		sellDetails.add(sale6);
		SellDetail sale7 = new SellDetail(new Date("11/17/2016"), projects.get(1), ce3);
		sellDetails.add(sale7);
		SellDetail sale8 = new SellDetail(new Date("11/18/2016"), projects.get(1), ce4);
		sellDetails.add(sale8);
		SellDetail sale9 = new SellDetail(new Date("11/21/2016"), projects.get(2), ce1);
		sellDetails.add(sale9);
		SellDetail sale10 = new SellDetail(new Date("11/22/2016"), projects.get(2), ce2);
		sellDetails.add(sale10);
		SellDetail sale11 = new SellDetail(new Date("11/23/2016"), projects.get(2), ce3);
		sellDetails.add(sale11);
		SellDetail sale12 = new SellDetail(new Date("11/24/2016"), projects.get(2), ce4);
		sellDetails.add(sale12);
		SellDetail sale13 = new SellDetail(new Date("11/26/2016"), projects.get(3), ce1);
		sellDetails.add(sale13);
		SellDetail sale14 = new SellDetail(new Date("11/27/2016"), projects.get(3), ce2);
		sellDetails.add(sale14);
		SellDetail sale15 = new SellDetail(new Date("11/28/2016"), projects.get(3), ce3);
		sellDetails.add(sale15);
		SellDetail sale16 = new SellDetail(new Date("11/29/2016"), projects.get(3), ce4);
		sellDetails.add(sale16);

		// ----------------------------------------------
		// some method calls
		for (InOut inout : inOuts) {
			inout.getEmployee().eachInOutPutInArray(inout);
		}
		for (Project project : projects) {
			project.getProductManager().eachProjectPutInArray(project);
			project.getTechnicalManager().eachProjectPutInArray(project);
			project.getCommercialManager().eachProjectPutInArray(project);
			for (ProductEmployee productEmployee : project.getProductEmployees()) {
				productEmployee.eachProjectPutInArray(project);
			}
		}
		for (SellDetail sellDetail : sellDetails) {
			sellDetail.getCommercialEmployee().eachSellingPutInArray(sellDetail);
			sellDetail.getProject().getSaleManager().eachSellingPutInArray(sellDetail);
			sellDetail.getProject().oneMoreSell();
		}
		for (Employee e : allEmployees) {
			e.calcBaseSalary();
			e.calcExtraHoursPay();
			e.calcInsurance();
			e.calcReward();
			e.calcImpureIncome();
			e.calcTax();
			e.calcPureIncome();
		}
		// ----------------------------------------------
	}

	// --------------------------------------------------------------
	// Accessors
	public ArrayList<Employee> getAllEmployees() {
		return allEmployees;
	}

	public ArrayList<ProductManager> getProductManagers() {
		return productManagers;
	}

	public ArrayList<TechnicalManager> getTechnicalManagers() {
		return technicalManagers;
	}

	public ArrayList<ProductEmployee> getProductEmployees() {
		return productEmployees;
	}

	public ArrayList<CommercialManager> getCommercialManagers() {
		return commercialManagers;
	}

	public ArrayList<SaleManager> getSaleManagers() {
		return saleManagers;
	}

	public ArrayList<CommercialEmployee> getCommercialEmployees() {
		return commercialEmployees;
	}

	public ArrayList<InOut> getInOuts() {
		return inOuts;
	}

	public ArrayList<Project> getProjects() {
		return projects;
	}

	public ArrayList<SellDetail> getSellDetails() {
		return sellDetails;
	}

	// -------------------------------------------------------------------------
	// Mutators
	public void setAllEmployees(ArrayList<Employee> allEmployees) {
		this.allEmployees = allEmployees;
	}

	public void setProductManagers(ArrayList<ProductManager> productManagers) {
		this.productManagers = productManagers;
	}

	public void setTechnicalManagers(ArrayList<TechnicalManager> technicalManagers) {
		this.technicalManagers = technicalManagers;
	}

	public void setProductEmployees(ArrayList<ProductEmployee> productEmployees) {
		this.productEmployees = productEmployees;
	}

	public void setCommercialManagers(ArrayList<CommercialManager> commercialManagers) {
		this.commercialManagers = commercialManagers;
	}

	public void setSaleManagers(ArrayList<SaleManager> saleManagers) {
		this.saleManagers = saleManagers;
	}

	public void setCommercialEmployees(ArrayList<CommercialEmployee> commercialEmployees) {
		this.commercialEmployees = commercialEmployees;
	}

	public void setInOuts(ArrayList<InOut> inOuts) {
		this.inOuts = inOuts;
	}

	public void setProjects(ArrayList<Project> projects) {
		this.projects = projects;
	}

	public void setSellDetails(ArrayList<SellDetail> sellDetails) {
		this.sellDetails = sellDetails;
	}
	// -------------------------------------------------------------------------
}
