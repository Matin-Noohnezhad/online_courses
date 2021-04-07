package org.bihe.DAO;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.bihe.IO.IOFile;
import org.bihe.bean.Person;
import org.bihe.bean.Position;
import org.bihe.bean.User;

public class UserPassDAO {

	// the key of hashmap is username.
	private static HashMap<String, Person> userPassList;
	private static UserPassDAO instance;
	private static final String fileName = "file.txt";

	private UserPassDAO() {

	}

	public static UserPassDAO getInstance() {
		if (instance == null) {
			instance = new UserPassDAO();
			return instance;
		}
		return instance;

	}

	@SuppressWarnings("unchecked")
	public void restore() {
		if (userPassList == null) {
			Object obj = IOFile.readObject(fileName);
			if (obj != null) {
				userPassList = (HashMap<String, Person>) obj;
			} else {
				userPassList = new HashMap<String, Person>();
			}
		}
	}

	public boolean store() {
		File f = new File(fileName);
		try {
			f.createNewFile();
			return IOFile.writeObject(fileName, userPassList);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return false;
	}

	public boolean addUserPassToList(String username, Person person) {
		if (userPassList == null) {
			restore();
		}
		userPassList.put(username, person);
		boolean b = store();
		return b;
	}

	public void addAdminAsAFirstOne() {
		restore();
		boolean b = false;
		if(userPassList.isEmpty()){
		User admin = new User("Matin", "Noohnezhad", "3430183485", "BandarLengeh", "admin", "admin", Position.ADMIN);
		addUserPassToList(admin.getUsername(), admin);
		b = true;
		}
		if(b == false){
		store();
		}
	}
	public Position checkUserPass(String username , String password){
		Position position = null;
		restore();
		Iterator<String> iter1 = userPassList.keySet().iterator();
		Iterator<Person> iter2 = userPassList.values().iterator();
		while(iter1.hasNext()){
			Person person = iter2.next();
			if(iter1.next().equals(username)){
				if(person.getPassword().equals(password)){
					position = person.getPosition();
				}
			}
		}
		store();
		return position;
	}

}
