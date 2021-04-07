package org.bihe;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class MyFrame extends JFrame implements ActionListener {

	private static final long serialVersionUID = -7107449472868737133L;
	private JButton returnButton = new JButton("بازگشت");
	private JButton enrollButton = new JButton("ثبت نام");
	private JButton entryButton = new JButton("ورود");
	private JButton enrollButton2 = new JButton("ثبت نام");
	private static int ret = 0;
	private static int enr = 0;
	private static int ent = 0;
	private static int enr2 = 0;
	private static final int WIDTH = 800;
	private static final int HEIGHT = 500;

	public MyFrame() {
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		setBounds(dim.width / 2 - WIDTH / 2, dim.height / 2 - HEIGHT / 2, WIDTH, HEIGHT);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		//
		setIconImage(Resource.getImage("pic3.png"));
		//
		setLayout(null);
		//
		this.setTitle("لاگین شرکت پست آسمان");

		add(getLoginPanel());
		//

	}

	@Override
	public void actionPerformed(ActionEvent e) {
		switch (e.getActionCommand()) {
		case "Enrollment":
//			setSize(WIDTH, HEIGHT);
			setTitle("ثبت نام کاربر");
			setContentPane(getEnrollmentPanel());

			break;

		case "Entry":
			JOptionPane.showMessageDialog(this, "به صفحه مورد نطر وارد شدید");
			break;
		case "Return":

//			setSize(WIDTH, HEIGHT);
			setTitle("لاگین شرکت پست آسمان");
			setContentPane(getLoginPanel());
			break;
		case "Enrollment2":
			JOptionPane.showMessageDialog(this, "شما با موفقیت ثبت نام شدین :)");

		default:

			break;
		}
	}

	private JPanel getLoginPanel() {
		JPanel loginPanel = new JPanel();
		//
		loginPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK, 7));
		loginPanel.setBackground(Color.WHITE);
		loginPanel.setLayout(null);
		//
		JLabel postOfficeName = new JLabel("پست آسمان");
		postOfficeName.setFont(postOfficeName.getFont().deriveFont(postOfficeName.getFont().getSize() * 1.6F));
		postOfficeName.setFont(postOfficeName.getFont().deriveFont(Font.BOLD));
		postOfficeName.setBounds(75, 10, 180, 23);
		loginPanel.add(postOfficeName);
		//
		JLabel username = new JLabel("نام کاربری");
		username.setFont(username.getFont().deriveFont(username.getFont().getSize() * 1.6F));
		username.setFont(username.getFont().deriveFont(Font.BOLD));
		username.setBounds(155, 65, 135, 23);
		loginPanel.add(username);
		//
		JTextField usernameInput = new JTextField();
		usernameInput.setBounds(10, 60, 135, 33);
		usernameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		loginPanel.add(usernameInput);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(165, 115, 135, 23);
		loginPanel.add(password);
		//
		JTextField passwordInput = new JTextField();
		passwordInput.setBounds(10, 110, 135, 33);
		passwordInput.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		loginPanel.add(passwordInput);
		//
		entryButton.setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		if (ent == 0) {
			entryButton.setFont(entryButton.getFont().deriveFont(entryButton.getFont().getSize() * 1.6F));
			ent += 1;
		}
		entryButton.setFont(entryButton.getFont().deriveFont(Font.BOLD));
		entryButton.setBounds(135, 260, 95, 45);
		entryButton.setBackground(Color.WHITE);
		entryButton.setContentAreaFilled(false);
		entryButton.setOpaque(true);
		entryButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		loginPanel.add(entryButton);
		//
		enrollButton.setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		if (enr == 0) {
			enrollButton.setFont(enrollButton.getFont().deriveFont(enrollButton.getFont().getSize() * 1.6F));
			enr += 1;
		}
		enrollButton.setFont(enrollButton.getFont().deriveFont(Font.BOLD));
		enrollButton.setBounds(30, 260, 95, 45);
		enrollButton.setBackground(Color.WHITE);
		enrollButton.setContentAreaFilled(false);
		enrollButton.setOpaque(true);
		enrollButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		loginPanel.add(enrollButton);
		//
		enrollButton.addActionListener(this);
		enrollButton.setActionCommand("Enrollment");
		loginPanel.add(enrollButton);
		entryButton.addActionListener(this);
		entryButton.setActionCommand("Entry");
		loginPanel.add(entryButton);
		//
		loginPanel.setBounds(0, 0, 272, 324);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		setBounds(dim.width / 2 - 291 / 2, dim.height / 2 - 372 / 2,291,372);
		//
		return loginPanel;
		//
	}

	// ---------------------------------------
	private JPanel getEnrollmentPanel() {
		JPanel enrollmentPanel = new JPanel();
		//
		enrollmentPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		enrollmentPanel.setBackground(Color.WHITE);
		enrollmentPanel.setLayout(null);
		//
		JLabel username = new JLabel("نام کاربری");
		username.setFont(username.getFont().deriveFont(username.getFont().getSize() * 1.6F));
		username.setFont(username.getFont().deriveFont(Font.BOLD));
		username.setBounds(580, 50, 120, 23);
		enrollmentPanel.add(username);
		//
		JTextField usernameInput = new JTextField();
		usernameInput.setBounds(410, 45, 150, 33);
		usernameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(usernameInput);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(595, 110, 120, 23);
		enrollmentPanel.add(password);
		//
		JTextField passwordInput = new JTextField();
		passwordInput.setBounds(410, 105, 150, 33);
		passwordInput.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(passwordInput);
		//
		JLabel cityName = new JLabel("شهر");
		cityName.setFont(cityName.getFont().deriveFont(cityName.getFont().getSize() * 1.6F));
		cityName.setFont(cityName.getFont().deriveFont(Font.BOLD));
		cityName.setBounds(633, 170, 120, 23);
		enrollmentPanel.add(cityName);
		//
		JTextField cityInput = new JTextField();
		cityInput.setBounds(410, 165, 150, 33);
		cityInput.setHorizontalAlignment(SwingConstants.RIGHT);
		cityInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(cityInput);
		//
		JLabel firstName = new JLabel("نام");
		firstName.setFont(firstName.getFont().deriveFont(firstName.getFont().getSize() * 1.6F));
		firstName.setFont(firstName.getFont().deriveFont(Font.BOLD));
		firstName.setBounds(295, 50, 120, 23);
		enrollmentPanel.add(firstName);
		//
		JTextField firstNameInput = new JTextField();
		firstNameInput.setBounds(30, 45, 150, 33);
		firstNameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		firstNameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(firstNameInput);
		//
		JLabel lastName = new JLabel("نام خانوادگی");
		lastName.setFont(lastName.getFont().deriveFont(lastName.getFont().getSize() * 1.6F));
		lastName.setFont(lastName.getFont().deriveFont(Font.BOLD));
		lastName.setBounds(195, 110, 150, 23);
		enrollmentPanel.add(lastName);
		//
		JTextField lastNameInput = new JTextField();
		lastNameInput.setBounds(30, 105, 150, 33);
		lastNameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		lastNameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(lastNameInput);
		//
		JLabel nationalCode = new JLabel("کد ملی");
		nationalCode.setFont(nationalCode.getFont().deriveFont(nationalCode.getFont().getSize() * 1.6F));
		nationalCode.setFont(nationalCode.getFont().deriveFont(Font.BOLD));
		nationalCode.setBounds(248, 170, 120, 23);
		enrollmentPanel.add(nationalCode);
		//
		JTextField nationalCodeInput = new JTextField();
		nationalCodeInput.setBounds(30, 165, 150, 33);
		nationalCodeInput.setHorizontalAlignment(SwingConstants.RIGHT);
		nationalCodeInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(nationalCodeInput);
		//
		enrollButton2.setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		if (enr2 == 0) {
			enrollButton2.setFont(enrollButton2.getFont().deriveFont(enrollButton2.getFont().getSize() * 1.6F));
			enr2 += 1;
		}
		enrollButton2.setFont(enrollButton2.getFont().deriveFont(Font.BOLD));
		enrollButton2.setBounds(395, 280, 110, 55);
		enrollButton2.setBackground(Color.WHITE);
		enrollButton2.setContentAreaFilled(false);
		enrollButton2.setOpaque(true);
		enrollButton2.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		enrollmentPanel.add(enrollButton2);
		//
		returnButton.setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		if (ret == 0) {
			returnButton.setFont(returnButton.getFont().deriveFont(returnButton.getFont().getSize() * 1.6F));
			ret += 1;
		}
		returnButton.setFont(returnButton.getFont().deriveFont(Font.BOLD));
		returnButton.setBounds(230, 280, 110, 55);
		returnButton.setBackground(Color.WHITE);
		returnButton.setContentAreaFilled(false);
		returnButton.setOpaque(true);
		returnButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		enrollmentPanel.add(returnButton);
		//
		returnButton.addActionListener(this);
		returnButton.setActionCommand("Return");
		enrollButton2.addActionListener(this);
		enrollButton2.setActionCommand("Enrollment2");
		enrollmentPanel.add(returnButton);
		enrollmentPanel.add(enrollButton2);
		//
		enrollmentPanel.setBounds(10, 10, 738, 363);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		setBounds(dim.width / 2 - 748 / 2, dim.height / 2 - 403 / 2,748,403);
		setSize(748, 403);
		//
		return enrollmentPanel;
		//
	}

}
