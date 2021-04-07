package org.bihe;

import java.io.*;

public class FileIO {

    public static String readFile(String filePath) {
        String output = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filePath));
            String line = null;
            while ((line = br.readLine()) != null) {
                if (output == null) {
                    output = line;
                } else
                    output = output +" "+ line;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        return output;
    }

}
