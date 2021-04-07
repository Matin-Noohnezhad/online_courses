package org.bihe.bean;

public class Employee extends Person {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6974680500221372755L;

	public Employee(String firstName, String lastName, String nationalCode, String city, String username, String password,
			Position position) {
		super(firstName, lastName, nationalCode, city, username, password, position);
	}

}
