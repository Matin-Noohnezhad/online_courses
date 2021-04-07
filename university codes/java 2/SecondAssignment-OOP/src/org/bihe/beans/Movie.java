package org.bihe.beans;

public class Movie {
	private String name;
	private Genre genre;
	private int count;
	private float price;
	private int ID;
	private static int basedNo = 1;
	private boolean hasEverBeenBorrowed;

	// ----------------------------------
	// constructor
	public Movie(String name, Genre genre, int count, float price) {
		this.name = name;
		this.genre = genre;
		this.count = count;
		this.price = price;
		this.ID = basedNo;
		basedNo++;
	}

	// ---------------------------------
	// Accessors
	public String getName() {
		return this.name;
	}

	public Genre getGenre() {
		return this.genre;
	}

	public int getCount() {
		return this.count;
	}

	public float getPrice() {
		return this.price;
	}

	public int getID() {
		return this.ID;
	}

	public boolean getHasEverBeenBorrowed() {
		return this.hasEverBeenBorrowed;
	}

	// ------------------------------------
	// Mutators
	public void setName(String name) {
		this.name = name;
	}

	public void setGenre(Genre genre) {
		this.genre = genre;
	}

	public void getCount(int count) {
		this.count = count;
	}

	public void getPrice(float price) {
		this.price = price;
	}

	public void setHasEverBeenBorrowed(boolean status) {
		this.hasEverBeenBorrowed = status;
	}
	// ------------------------------------
	// override
	//equals method
	@Override
	public boolean equals(Object o) {
		if (o.equals(null)) {
			return false;
		}

		if (o instanceof Movie) {
			Movie m = (Movie) o;
			if (this.genre.equals(m.getGenre()) && this.name.equals(m.getName()) && this.count == m.count
					&& this.price == m.price && this.ID == m.ID) {
				return true;
			}
		}
		return false;
	}
	//------------------------------------
	//toString method
	public String toString(){
		return this.name+"("+this.genre+")";
	}

}
