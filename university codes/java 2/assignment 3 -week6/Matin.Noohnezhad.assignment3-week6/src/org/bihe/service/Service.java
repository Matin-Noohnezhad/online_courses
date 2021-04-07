package org.bihe.service;

import java.util.ArrayList;

import org.bihe.beans.CommercialEmployee;
import org.bihe.beans.CommercialManager;
import org.bihe.beans.Data;
import org.bihe.beans.Employee;
import org.bihe.beans.InOut;
import org.bihe.beans.ProductEmployee;
import org.bihe.beans.ProductManager;
import org.bihe.beans.Project;
import org.bihe.beans.SaleManager;
import org.bihe.beans.SellDetail;
import org.bihe.beans.TechnicalManager;

public class Service {
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

	// --------------------------------------------------------------
	public void added() {
		Data data = new Data();
		data.data();
		allEmployees.addAll(data.getAllEmployees());
		productManagers.addAll(data.getProductManagers());
		technicalManagers.addAll(data.getTechnicalManagers());
		productEmployees.addAll(data.getProductEmployees());
		commercialManagers.addAll(data.getCommercialManagers());
		saleManagers.addAll(data.getSaleManagers());
		commercialEmployees.addAll(data.getCommercialEmployees());
		inOuts.addAll(data.getInOuts());
		projects.addAll(data.getProjects());
		sellDetails.addAll(data.getSellDetails());
	}

	// -------------------------------------------------------------
	public void Question1() {
		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 1 is:");
		System.out.println("the impure income of each employee is:");
		for (Employee employee : allEmployees) {
			System.out.print(employee.toString() + ": ");
			System.out.println(employee.getImpureIncome());
		}
		System.out.println("-------------------------------------------------------------------");
	}

	// ----------------------------------------
	public void Question2() {
		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 2 is:");
		System.out.println("the pure income of each employee is:");
		for (Employee employee : allEmployees) {
			System.out.print(employee.toString() + ": ");
			System.out.println(employee.getPureIncome());
		}
		System.out.println("-------------------------------------------------------------------");
	}

	// ----------------------------------------
	public void Question3() {
		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 3 is:");
		System.out.println("the reward of each employee is:");
		for (Employee employee : allEmployees) {
			System.out.print(employee.toString() + ": ");
			System.out.println(employee.getRewardPrice());
		}
		System.out.println("---------------------------");
		System.out.println("the extra hours pay of each employee is:");
		for (Employee employee : allEmployees) {
			System.out.print(employee.toString() + ": ");
			System.out.println(employee.getExtraHoursPay());
		}
		System.out.println("-------------------------------------------------------------------");
	}

	// ----------------------------------------
	public void Question4() {
		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 4 is:");
		System.out.println("the sum of hours for each employee is:");
		for (Employee employee : allEmployees) {
			int sum = 0;
			for (InOut inOut : employee.getWorkHours()) {
				sum += inOut.getSumOfHours();
			}
			System.out.print(employee.toString() + ": ");
			System.out.println(sum);
		}
		System.out.println("---------------------------");
		System.out.println("the work hours of every day for each employee is:");
		for (Employee employee : allEmployees) {
			System.out.print(employee.toString() + ": ");
			for (InOut inOut : employee.getWorkHours()) {
				System.out.print(inOut.getSumOfHours() + " - ");
			}
			System.out.println();
		}
		System.out.println("-------------------------------------------------------------------");
	}
	public void Question5(){

		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 5 is:");
		System.out.println("the projects and employees which help to make it is:");
		for(Project project : projects){
			System.out.println(project.toString());
		}
		System.out.println("-------------------------------------------------------------------");
	}
	public void Question6(){


		if (allEmployees.size() == 0) {
			added();
		}
		// -----------
		System.out.println("-------------------------------------------------------------------");
		System.out.println("The answer of question 6 is:");
		System.out.println("the projects that sells and employees which help to sell it is:");
		for(SellDetail sellDetail: sellDetails){
			System.out.println(sellDetail.toString());
		}
		System.out.println("-------------------------------------------------------------------");
	}
}
