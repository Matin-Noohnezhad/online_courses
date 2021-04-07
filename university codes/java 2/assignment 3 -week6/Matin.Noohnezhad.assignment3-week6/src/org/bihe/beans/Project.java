package org.bihe.beans;

import java.util.ArrayList;
import java.util.Date;

public class Project {
	private String name;
	private Date start;
	private Date finish;
	private CommercialManager commercialManager;
	private ProductManager productManager;
	private TechnicalManager technicalManager;
	private ArrayList<ProductEmployee> productEmployees = new ArrayList<>();
	private SaleManager saleManager;
	private float price;
	private int sellCount;
	private ArrayList<SellDetail> sells;

	// ------------------------------------
	// constructor
	public Project(String name, Date start, Date finish, CommercialManager commercialManager,
			ProductManager productManager, TechnicalManager technicalManager,
			ArrayList<ProductEmployee> productEmployees, SaleManager saleManager, float price) {
		this.name = name;
		this.start = start;
		this.finish = finish;
		this.commercialManager = commercialManager;
		this.productManager = productManager;
		this.technicalManager = technicalManager;
		this.productEmployees = productEmployees;
		this.saleManager = saleManager;
		this.price = price;
		this.sellCount = 0;
	}

	// -------------------------------------
	// method
	public void oneMoreSell() {
		this.sellCount += 1;
	
	}

	// -------------------------------------
	// Accessors
	public String getName() {
		return name;
	}

	public Date getStart() {
		return start;
	}

	public Date getFinish() {
		return finish;
	}

	public CommercialManager getCommercialManager() {
		return commercialManager;
	}

	public ProductManager getProductManager() {
		return productManager;
	}

	public TechnicalManager getTechnicalManager() {
		return technicalManager;
	}

	public ArrayList<ProductEmployee> getProductEmployees() {
		return productEmployees;
	}

	public SaleManager getSaleManager() {
		return saleManager;
	}

	public float getPrice() {
		return price;
	}

	public int getSellCount() {
		return sellCount;
	}

	public ArrayList<SellDetail> getSells() {
		return sells;
	}

	// ----------------------------------------
	// Mutators
	public void setName(String name) {
		this.name = name;
	}

	public void setStart(Date start) {
		this.start = start;
	}

	public void setFinish(Date finish) {
		this.finish = finish;
	}

	public void setCommercialManager(CommercialManager commercialManager) {
		this.commercialManager = commercialManager;
	}

	public void setProductManager(ProductManager productManager) {
		this.productManager = productManager;
	}

	public void setTechnicalManager(TechnicalManager technicalManager) {
		this.technicalManager = technicalManager;
	}

	public void setProductEmployees(ArrayList<ProductEmployee> productEmployees) {
		this.productEmployees = productEmployees;
	}

	public void setSaleManager(SaleManager saleManager) {
		this.saleManager = saleManager;
	}

	public void setPrice(float price) {
		this.price = price;
	}

	public void setSellCount(int sellCount) {
		this.sellCount = sellCount;
	}

	public void setSells(ArrayList<SellDetail> sells) {
		this.sells = sells;
	}

	// ----------------------------------------------------------------------
	// override toString method
	public String toString() {
		String string = "";
		for (ProductEmployee p : productEmployees) {
			string += p.toString() + " - ";
		}
		return "project name: " + this.name + " - commercial manager of this project is: "
				+ this.commercialManager.toString() + " - product manager of this project is: "
				+ this.productManager.toString() + " - technical manager of this project is: "
				+ this.technicalManager.toString() + " - the product employees of this project are: " + string;
	}
	// -----------------------------------------------------------------------
}
