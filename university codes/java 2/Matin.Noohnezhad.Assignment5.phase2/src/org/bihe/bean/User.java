package org.bihe.bean;

public class User extends Person {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5964819207760917570L;

	public User(String firstName, String lastName, String nationalCode, String city, String username, String password,
			Position position) {
		super(firstName, lastName, nationalCode, city, username, password, position);
	}

}
