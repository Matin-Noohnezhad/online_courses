package org.bihe.IO;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class IOFile {

	public static boolean writeObject(String fileName, Serializable s) {

		try {
			ObjectOutputStream objout = new ObjectOutputStream(new FileOutputStream(fileName));
			objout.writeObject(s);
			objout.flush();
			objout.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return false;
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static Serializable readObject(String fileName) {
		Serializable object = null;
		try {
			//to check if the file exists
			File f = new File(fileName);
			if(!f.exists()){
				f.createNewFile();
			}
			//to check that the file is not empty
			BufferedReader br = new BufferedReader(new FileReader(fileName));     
			if (br.readLine() != null) {
				ObjectInputStream objin = new ObjectInputStream(new FileInputStream(fileName));
				object = (Serializable) objin.readObject();
				objin.close();
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		return object;

	}

}
