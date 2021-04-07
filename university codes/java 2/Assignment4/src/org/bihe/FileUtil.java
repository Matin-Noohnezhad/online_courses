package org.bihe;

import java.awt.Desktop;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.Scanner;

/**
 * 
 * @author Farzam This class provides some utilities based on file operations.
 */
public class FileUtil {

	/**
	 * Comment here
	 * 
	 * @param sourcePath
	 * @param destinationPath
	 * @return
	 */
	public static Log copyByIO(String sourcePath, String destinationPath) {
		// TODO
		// Copy file or folder from sourcePath to destinationPath using IO, then
		// return a
		// log object that contains information of this procedure.

		// Your code
		BufferedInputStream bufIn = null;
		BufferedOutputStream bufOut = null;
		long start = 0;
		long finish = 0;
		Date date = new Date();
		ArrayList<String> moreInfo = new ArrayList<>();
		moreInfo.add("source path is: " + sourcePath);
		moreInfo.add("\ndestination path is: " + destinationPath);
		try {
			bufIn = new BufferedInputStream(new FileInputStream(sourcePath));
			bufOut = new BufferedOutputStream(new FileOutputStream(destinationPath));
			int r;
			start = System.currentTimeMillis();
			while ((r = bufIn.read()) != -1) {
				bufOut.write(r);
			}
			bufOut.flush();
			finish = System.currentTimeMillis();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				bufIn.close();
			} catch (IOException e) {
			}
			try {
				bufOut.close();
			} catch (IOException e) {
			}
		}
		Log log = new Log(date, (finish - start), MethodNames.COPY_BY_IO, moreInfo);
		return log;
	}

	/**
	 * Comment here
	 * 
	 * @param sourcePath
	 * @param destinationPath
	 * @return
	 */
	public static Log copyByNIO(String sourcePath, String destinationPath) {
		// TODO
		// Copy file or folder from sourcePath to destinationPath using NIO,
		// then return a
		// log object that contains information of this procedure.

		// Your code
		FileChannel fcin = null;
		FileChannel fcout = null;
		FileInputStream fis = null;
		FileOutputStream fos = null;
		long start = 0;
		long finish = 0;
		Date date = new Date();
		ArrayList<String> moreInfo = new ArrayList<>();
		moreInfo.add("source path is: " + sourcePath);
		moreInfo.add("\ndestination path is: " + destinationPath);

		try {
			fis = new FileInputStream(sourcePath);
			fos = new FileOutputStream(destinationPath);
			fcin = fis.getChannel();
			fcout = fos.getChannel();
			ByteBuffer buffer = ByteBuffer.allocate(1024);
			start = System.currentTimeMillis();
			while (true) {
				buffer.clear();
				int r = fcin.read(buffer);
				if (r == -1) {
					break;
				}
				buffer.flip();
				fcout.write(buffer);
			}
			finish = System.currentTimeMillis();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				fcin.close();
			} catch (IOException e) {
			}
			try {
				fcout.close();
			} catch (IOException e) {
			}
		}

		Log log = new Log(date, (finish - start), MethodNames.COPY_BY_NIO, moreInfo);
		return log;
	}

	/**
	 * Comment here
	 * 
	 * @param musicSourcePath
	 * @return
	 */
	public static Log playMusic(String musicSourcePath) {
		// TODO
		// Play a music file with the default player of OS.

		// Your code
		File songFile = new File(musicSourcePath);
		ArrayList<String> musics = new ArrayList<>();
		musics.add("music name: " + songFile.getName());
		Date date = new Date();
		long start = 0;
		long finish = 0;
		try {
			start = System.currentTimeMillis();
			if (songFile.getName().endsWith("mp3")) {
				Desktop.getDesktop().open(songFile);
			}
			finish = System.currentTimeMillis();
		} catch (IOException e) {
			e.printStackTrace();
		}

		Log log = new Log(date, (finish - start), MethodNames.PLAY_MUSIC, musics);
		return log;
	}

	/**
	 * Comment here
	 * 
	 * @param path
	 * @return
	 */
	public static Log calculateSize(String path) {
		// TODO
		// Calculate size of path (folder or file), then add the size with the
		// format below to the "moreInfo" field of Log object.
		// x GB and y MB z KB

		// Your code
		File file = new File(path);
		long start = 0;
		long finish = 0;
		ArrayList<String> moreInfo = new ArrayList<>();
		start = System.currentTimeMillis();
		moreInfo.add("the path of file is: " + path);
		long size = file.length();
		finish = System.currentTimeMillis();
		long GB = size / (1024 * 1024 * 1024);
		long MB = (size % (1024 * 1024 * 1024)) / (1024 * 1024);
		long KB = ((size % (1024 * 1024 * 1024)) % (1024 * 1024)) / 1024;
		moreInfo.add("\nThe size of the file is: " + GB + " GB and " + MB + " MB " + KB + " KB");
		Log log = new Log(new Date(), (finish - start), MethodNames.CALCULATE_SIZE, moreInfo);
		return log;
	}

	/**
	 * Comment here
	 * 
	 * @param folderPath
	 * @return
	 */
	public static Log picturesList(String folderPath) {
		// TODO
		// Find pictures with [JPEG,JPG,PNG,GIF,TIFF,BMP] formats in
		// "folderPath", and add name of the pictures to moreInfo

		// Your code
		File directory = new File(folderPath);
		File[] files = directory.listFiles();
		ArrayList<String> moreInfo = new ArrayList<>();
		long start = 0;
		long finish = 0;
		start = System.currentTimeMillis();
		for (File file : files) {
			if (file.getName().endsWith("JPEG") || file.getName().endsWith("JPG") || file.getName().endsWith("PNG")
					|| file.getName().endsWith("GIF") || file.getName().endsWith("TIFF")
					|| file.getName().endsWith("BMP")) {
				moreInfo.add(file.getName());
			}
		}
		finish = System.currentTimeMillis();
		Log log = new Log(new Date(), (finish - start), MethodNames.PICTURES_LIST, moreInfo);
		return log;
	}

	/**
	 * Comment here
	 * 
	 * @param logList
	 * @param fileName
	 * @return
	 */
	public static boolean writeLogsTxt(ArrayList<Log> logList, String fileName) {
		// TODO
		// Write "logList" objects to a text file
		// each Log object should be in the format below:
		// yyyyy-mm-dd hh:mm:ss,timeOfAction,methodCall,moreInfo(the toString
		// method of moreInfo)

		// Your code

		File file = new File(fileName);
		StringBuffer list = new StringBuffer();
		String line = System.getProperty("line.separator");

		for (Log log : logList) {
			list.append(log.getStartActionDate().getYear() + "-" + log.getStartActionDate().getMonth() + "-"
					+ log.getStartActionDate().getDate() + " " + log.getStartActionDate().getHours() + ":"
					+ log.getStartActionDate().getMinutes() + ":" + log.getStartActionDate().getSeconds() + ","
					+ log.getTimeOfAction() + "," + log.getMethodCall() + "," + log.getMoreInfo().toString())
					.append(line);
		}
		PrintWriter pw = null;

		try {
			pw = new PrintWriter(file);
			pw.write(list.toString());
			pw.flush();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}

		return true;
	}

	/**
	 * Comment here
	 * 
	 * @param fileName
	 * @return
	 */
	public static ArrayList<Log> readLogsTxt(String fileName) {
		// TODO
		// Read from "fileName" text file
		// each line represent an Log object and, each line should be in the
		// format below:
		// yyyyy-mm-dd hh:mm:ss,timeOfAction,methodCall,moreInfo

		// Your code

		Scanner sc = null;
		int numberOfLine = 0;
		ArrayList<Log> logs = new ArrayList<>();
		try {
			sc = new Scanner(new BufferedReader(new FileReader(fileName)));
			while (sc.hasNextLine()) {
				numberOfLine++;
				sc.nextLine();
			}
			sc = new Scanner(new BufferedReader(new FileReader(fileName)));

			for (int i = 0; i < numberOfLine; i++) {
				Scanner sc2 = new Scanner(sc.nextLine());
				sc2.useDelimiter("-");
				int year = Integer.parseInt(sc2.next());
				int month = Integer.parseInt(sc2.next());
				sc2 = new Scanner(sc2.next());
				sc2.useDelimiter(" ");
				int day = Integer.parseInt(sc2.next());
				sc2 = new Scanner(sc2.next());
				sc2.useDelimiter(":");
				int hour = Integer.parseInt(sc2.next());
				int minute = Integer.parseInt(sc2.next());
				sc2 = new Scanner(sc2.next());
				sc2.useDelimiter(",");
				int second = Integer.parseInt(sc2.next());
				Date date = new Date(year, month, day, hour, minute, second);
				Long timeOfAction = Long.parseLong(sc2.next());
				MethodNames methodName = MethodNames.valueOf(sc2.next());
				ArrayList<String> moreInfo = new ArrayList<>();
				int n = 0;
				while (sc2.hasNext()) {
					if (n == 0) {
						String string = sc2.next();
						
						//substring(1,...) for delete "["
						moreInfo.add(string.substring(1, string.length()));
						n++;
					} else {
						if (sc2.hasNext()) {
							moreInfo.add(sc2.next());
						} else {
							String string = sc2.next();
							
							//substring(...,string.length() - 1) for delete "]" 
							moreInfo.add(string.substring(0, string.length() - 1));
						}
					}
				}
				Log log = new Log(date, timeOfAction, methodName, moreInfo);
				logs.add(log);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}finally{
			sc.close();
		}

		return logs;
	}

	/**
	 * Comment here
	 * 
	 * @param logList
	 * @param fileName
	 * @return
	 */
	public static boolean writeLogsObj(ArrayList<Log> logList, String fileName) {
		// TODO
		// Write "logList" object to a file

		// Your code
		
		File file = new File(fileName);
		for(Log log : logList){
			ObjectOutputStream objOut = null;
			
			try {
				objOut = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
				objOut.writeObject(log);
				objOut.flush();
				objOut.writeUTF("\n");
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}finally{
				try {
					objOut.close();
				} catch (IOException e) {
				}
			}
		}

		return true;
	}

	/**
	 * Comment here
	 * 
	 * @param fileName
	 * @return
	 */
	public static ArrayList<Log> readLogsObj(String fileName) {
		// TODO
		// Read "logList" object from a file

		// Your code

		File file = new File(fileName);
		ObjectInputStream objIn = null;
		ArrayList<Log> logs = new ArrayList<>();
		
		try {
			Scanner sc = new Scanner(file);
			sc.useDelimiter("\n");
			while(sc.hasNext()){
			objIn = new ObjectInputStream(new BufferedInputStream(new FileInputStream(sc.next())));
			Log log = (Log) objIn.readObject();
			logs.add(log);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return logs;
	}

}
