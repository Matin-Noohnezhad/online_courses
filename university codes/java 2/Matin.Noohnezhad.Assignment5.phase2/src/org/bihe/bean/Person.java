package org.bihe.bean;

import java.io.Serializable;

public abstract class Person implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8939530050489830362L;
	private String firstName;
	private String lastName;
	private String nationalCode;
	private String city;
	private String username;
	private String password;
	private Position position;
	
	public Person(String firstName, String lastName,String nationalCode, String city, String username, String password,Position position) {
	
		this.firstName = firstName;
		this.lastName = lastName;
		this.nationalCode = nationalCode;
		this.city = city;
		this.username = username;
		this.password = password;
		this.position = position;
		
	}

	public String getFirstName() {
		return firstName;
	}

	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public String getNationalCode() {
		return nationalCode;
	}

	public void setNationalCode(String nationalCode) {
		this.nationalCode = nationalCode;
	}

	public String getCity() {
		return city;
	}

	public void setCity(String city) {
		this.city = city;
	}

	public String getUsername() {
		return username;
	}

	public void setUsername(String username) {
		this.username = username;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}
	public Position getPosition(){
		return position;
	}
	public void setPosition(Position position){
		this.position = position;
	}

	@Override
	public boolean equals(Object o) {
		Person person = (Person) o;
		if(firstName.equals(person.getFirstName())&&lastName.equals(person.getLastName())&&username.equals(person.getUsername())&&password.equals(person.getPassword())){
			return true;
		}else{
			return false;
		}
	}

	@Override
	public String toString() {
		return  firstName+" "+lastName;
	}
	
	
	

}
