package org.bihe;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;
import javax.swing.border.EmptyBorder;

public class MainFrame extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private ArrayList<Log> logs = new ArrayList<>();
	private JPanel contentPane;

	/**
	 * Create the frame.
	 */
	public MainFrame() {
		setSize(600, 400);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		this.setLocation(dim.width / 2 - this.getSize().width / 2, dim.height / 2 - this.getSize().height / 2);
		setResizable(false);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		JLabel label = new JLabel("Choose An Action");
		label.setFont(new Font("Tahoma", Font.PLAIN, 35));
		label.setBounds(152, 11, 301, 50);
		contentPane.add(label);

		JButton cpIO = new JButton("Copy by IO");
		cpIO.setBounds(50, 100, 150, 50);

		cpIO.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				File sourceFile = null, destFile = null;
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());

					JFileChooser chooserDest = new JFileChooser();
					chooserDest.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

					chooserDest.setSelectedFile(sourceFile);
					int returnValDest = chooserDest.showSaveDialog(MainFrame.this);
					if (returnValDest == JFileChooser.APPROVE_OPTION) {
						destFile = chooserDest.getSelectedFile();
						System.out.println(
								"You chose to save this file: " + chooserDest.getSelectedFile().getAbsolutePath());
					}
				}
				if (sourceFile != null && destFile != null) {
					Log log = FileUtil.copyByIO(sourceFile.getAbsolutePath(), destFile.getAbsolutePath());
					logs.add(log);
					ShowLogFrame showLog = new ShowLogFrame(log);
					showLog.setVisible(true);
				}
			}
		});
		contentPane.add(cpIO);

		JButton cpNIO = new JButton("Copy by NIO");
		cpNIO.setBounds(225, 100, 150, 50);
		cpNIO.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File sourceFile = null, destFile = null;
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());

					JFileChooser chooserDest = new JFileChooser();
					chooserDest.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

					chooserDest.setSelectedFile(sourceFile);
					int returnValDest = chooserDest.showSaveDialog(MainFrame.this);
					if (returnValDest == JFileChooser.APPROVE_OPTION) {
						destFile = chooserDest.getSelectedFile();
						System.out.println(
								"You chose to save this file: " + chooserDest.getSelectedFile().getAbsolutePath());
					}
				}
				if (sourceFile != null && destFile != null) {
					Log log = FileUtil.copyByNIO(sourceFile.getAbsolutePath(), destFile.getAbsolutePath());
					logs.add(log);
					ShowLogFrame showLog = new ShowLogFrame(log);
					showLog.setVisible(true);
				}
			}
		});
		contentPane.add(cpNIO);

		JButton playMusic = new JButton("Play music");
		playMusic.setBounds(400, 100, 150, 50);
		playMusic.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
				File sourceFile = null;
				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());
				}
				if (sourceFile != null) {
					Log l = FileUtil.playMusic(sourceFile.getAbsolutePath());
					ShowLogFrame slf = new ShowLogFrame(l);
					slf.setVisible(true);
					logs.add(l);
				}
			}
		});
		contentPane.add(playMusic);

		JButton calculateSize = new JButton("Calculate size");
		calculateSize.setBounds(50, 175, 150, 50);
		calculateSize.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

				File sourceFile = null;
				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());
				}
				if (sourceFile != null) {
					Log l = FileUtil.calculateSize(sourceFile.getAbsolutePath());
					ShowLogFrame slf = new ShowLogFrame(l);
					slf.setVisible(true);
					logs.add(l);
				}
			}
		});
		contentPane.add(calculateSize);

		JButton picturesList = new JButton("Pictures List");
		picturesList.setBounds(225, 175, 150, 50);
		picturesList.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

				File sourceFile = null;
				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());
				}
				if (sourceFile != null) {
					Log l = FileUtil.picturesList(sourceFile.getAbsolutePath());
					ShowLogFrame slf = new ShowLogFrame(l);
					slf.setVisible(true);
					logs.add(l);
				}
			}
		});
		contentPane.add(picturesList);

		JButton writeLogs = new JButton("Write logs(txt)");
		writeLogs.setBounds(400, 175, 150, 50);
		writeLogs.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File destFile = null;
				JFileChooser chooserDest = new JFileChooser();
				chooserDest.setFileSelectionMode(JFileChooser.FILES_ONLY);

				int returnValDest = chooserDest.showSaveDialog(MainFrame.this);
				if (returnValDest == JFileChooser.APPROVE_OPTION) {
					destFile = chooserDest.getSelectedFile();
					System.out
							.println("You chose to save this file: " + chooserDest.getSelectedFile().getAbsolutePath());
				}

				if (destFile != null) {
					boolean b = FileUtil.writeLogsTxt(logs, destFile.getAbsolutePath());
					if (b) {
						JOptionPane.showMessageDialog(null, "Writing logs finished successfully.",
								"Write Logs Txt Mode", JOptionPane.INFORMATION_MESSAGE);
					} else {
						JOptionPane.showMessageDialog(null, "Error detected while writing logs.", "Write Logs Txt Mode",
								JOptionPane.ERROR_MESSAGE);
					}
				}

			}
		});
		contentPane.add(writeLogs);

		JButton readLogs = new JButton("Read logs(txt)");
		readLogs.setBounds(50, 250, 150, 50);
		readLogs.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File sourceFile = null;
				JFileChooser chooser = new JFileChooser();
				chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());

				}
				if (sourceFile != null) {
					ArrayList<Log> readLogs = FileUtil.readLogsTxt(sourceFile.getAbsolutePath());
					LogsListFrame llf = new LogsListFrame(readLogs);
					llf.setVisible(true);
				}
			}
		});
		contentPane.add(readLogs);

		JButton writeLogsObj = new JButton("Write logs(obj)");
		writeLogsObj.setBounds(225, 250, 150, 50);
		writeLogsObj.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File destFile = null;
				JFileChooser chooserDest = new JFileChooser();
				chooserDest.setFileSelectionMode(JFileChooser.FILES_ONLY);

				int returnValDest = chooserDest.showSaveDialog(MainFrame.this);
				if (returnValDest == JFileChooser.APPROVE_OPTION) {
					destFile = chooserDest.getSelectedFile();
					System.out
							.println("You chose to save this file: " + chooserDest.getSelectedFile().getAbsolutePath());
				}

				if (destFile != null) {
					boolean b = FileUtil.writeLogsObj(logs, destFile.getAbsolutePath());
					if (b) {
						JOptionPane.showMessageDialog(null, "Writing logs finished successfully.",
								"Write Logs Object Mode", JOptionPane.INFORMATION_MESSAGE);
					} else {
						JOptionPane.showMessageDialog(null, "Error detected while writing logs.",
								"Write Logs Object Mode", JOptionPane.ERROR_MESSAGE);
					}
				}

			}
		});
		contentPane.add(writeLogsObj);

		JButton readLogsObj = new JButton("Read logs(obj)");
		readLogsObj.setBounds(400, 250, 150, 50);
		readLogsObj.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File sourceFile = null;
				JFileChooser chooser = new JFileChooser();
				int returnVal = chooser.showOpenDialog(MainFrame.this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					sourceFile = chooser.getSelectedFile();
					System.out.println("You chose to open this file: " + chooser.getSelectedFile().getAbsolutePath());

				}
				ArrayList<Log> readLogs = FileUtil.readLogsObj(sourceFile.getAbsolutePath());
				LogsListFrame llf = new LogsListFrame(readLogs);
				llf.setVisible(true);
			}
		});
		contentPane.add(readLogsObj);
	}

	public static void changeLaf(JFrame frame, String laf) {
		if (laf.equals("metal")) {
			try {
				UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
			} catch (ClassNotFoundException | InstantiationException | IllegalAccessException
					| UnsupportedLookAndFeelException e) {
				e.printStackTrace();
			}
		}
		if (laf.equals("nimbus")) {
			try {

				UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
			} catch (ClassNotFoundException | InstantiationException | IllegalAccessException
					| UnsupportedLookAndFeelException e) {
				e.printStackTrace();
			}
		}
		if (laf.equals("system")) {
			try {
				UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
			} catch (ClassNotFoundException | InstantiationException | IllegalAccessException
					| UnsupportedLookAndFeelException e) {
				e.printStackTrace();
			}
		}

		SwingUtilities.updateComponentTreeUI(frame);
	}
}
