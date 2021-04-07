package org.bihe;

import java.awt.Image;
import java.awt.Toolkit;

public class Resource {

	public static Image getImage(String name){
		return Toolkit.getDefaultToolkit().getImage(Resource.class.getResource(name));
	}
	
}
