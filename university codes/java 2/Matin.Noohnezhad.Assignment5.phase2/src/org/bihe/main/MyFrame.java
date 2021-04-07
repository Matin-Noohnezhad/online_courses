package org.bihe.main;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import org.bihe.DAO.UserPassDAO;
import org.bihe.bean.Position;
import org.bihe.bean.User;

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
	private JTextField usernameInput2;
	private JTextField passwordInput2;
	private JTextField cityInput;
	private JTextField firstNameInput;
	private JTextField lastNameInput;
	private JTextField nationalCodeInput;
	private JTextField usernameInput1;
	private JTextField passwordInput1;
	private JButton returnButton2;
	private JButton employeeButton2;
	private Position POSITION = Position.USER;

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
			// setSize(WIDTH, HEIGHT);
			setTitle("ثبت نام کاربر");
			setContentPane(getEnrollmentPanel());
			POSITION = Position.USER;

			break;

		case "Entry":
			// JOptionPane.showMessageDialog(this, "به صفحه مورد نطر وارد
			// شدید");
			String username2 = usernameInput1.getText();
			String password2 = passwordInput1.getText();
			UserPassDAO upd2 = UserPassDAO.getInstance();
			Position position2 = upd2.checkUserPass(username2, password2);
			if (position2 == null) {
				JOptionPane.showMessageDialog(this, "چنین نام کاربری و رمز عبوری در سایت وجود ندارد.", "خطا",
						JOptionPane.INFORMATION_MESSAGE);
			} else if (position2 == Position.ADMIN) {
				setExtendedState(JFrame.MAXIMIZED_BOTH);
				setTitle("سیستم مرسولات پستی آسمان");
				getContentPane().removeAll();
				//
				setLayout(new BorderLayout());
				add(getAdminPanel(), BorderLayout.CENTER);
				//
				JOptionPane.showMessageDialog(this, position2 + " was entered.");
			} else {

				setExtendedState(JFrame.MAXIMIZED_BOTH);
				setTitle("سیستم مرسولات پستی آسمان");
				getContentPane().removeAll();
				getContentPane().setBackground(Color.cyan);
				JOptionPane.showMessageDialog(this, position2 + " was entered.");
			}
			break;
		case "Return":

			// setSize(WIDTH, HEIGHT);
			setTitle("لاگین شرکت پست آسمان");
			setContentPane(getLoginPanel());
			break;
		case "Enrollment2":
			// JOptionPane.showMessageDialog(this, "شما با موفقیت ثبت نام شدین
			// :)");
			if (usernameInput2.getText().length() == 0 && passwordInput2.getText().length() == 0
					&& cityInput.getText().length() == 0 && firstNameInput.getText().length() == 0
					&& lastNameInput.getText().length() == 0 && nationalCodeInput.getText().length() == 0) {
				JOptionPane.showMessageDialog(this, "همه فیلدها باید پر باشد.", "خطا", JOptionPane.INFORMATION_MESSAGE);
			} else if (nationalCodeInput.getText().length() > 10) {
				JOptionPane.showMessageDialog(this, "تعداد ارقام کد ملی باید کمتر از 10 عدد باشد.", "خطا",
						JOptionPane.INFORMATION_MESSAGE);
			} else {
				String username = usernameInput2.getText();
				String password = passwordInput2.getText();
				String city = cityInput.getText();
				String firstName = firstNameInput.getText();
				String lastName = lastNameInput.getText();
				String nationalCode = nationalCodeInput.getText();
				Position position = POSITION;
				User user = new User(firstName, lastName, nationalCode, city, username, password, position);
				UserPassDAO upd = UserPassDAO.getInstance();
				boolean isAdded = upd.addUserPassToList(username, user);
				if (isAdded) {
					// JOptionPane.showMessageDialog(this, "کاربر جدید ثبت نام
					// شد.");
					int n = JOptionPane.showOptionDialog(this, "کاربر جدید ثبت نام شد.", "Message",
							JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE, null, null, null);
					if (n == 0) {
						usernameInput2.setText("");
						passwordInput2.setText("");
						cityInput.setText("");
						firstNameInput.setText("");
						lastNameInput.setText("");
						nationalCodeInput.setText("");
					}
				} else {
					JOptionPane.showMessageDialog(this, "کاربر جدید ثبت نام نشد.");
				}
			}
			break;
		case "Return2":
			// setSize(WIDTH, HEIGHT);
			setTitle("لاگین شرکت پست آسمان");
			setContentPane(getLoginPanel());

			break;
			
		case "Employee":
			// setSize(WIDTH, HEIGHT);
						setTitle("ثبت نام کارمند");
						setContentPane(getEnrollmentPanel());
						POSITION = Position.EMPLOYEE;

			break;

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
		usernameInput1 = new JTextField();
		usernameInput1.setBounds(10, 60, 135, 33);
		usernameInput1.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput1.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		usernameInput1.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (passwordInput1.getText().length() != 0 && usernameInput1.getText().length() != 0) {
					entryButton.setEnabled(true);
				} else {
					entryButton.setEnabled(false);
				}
			}
		});
		loginPanel.add(usernameInput1);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(165, 115, 135, 23);
		loginPanel.add(password);
		//
		passwordInput1 = new JTextField();
		passwordInput1.setBounds(10, 110, 135, 33);
		passwordInput1.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput1.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		passwordInput1.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (usernameInput1.getText().length() != 0 && passwordInput1.getText().length() != 0) {
					entryButton.setEnabled(true);
				} else {
					entryButton.setEnabled(false);
				}
			}
		});
		loginPanel.add(passwordInput1);
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
		entryButton.addActionListener(this);
		entryButton.setActionCommand("Entry");
		entryButton.setEnabled(false);
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
		enrollButton.addActionListener(this);
		enrollButton.setActionCommand("Enrollment");
		loginPanel.add(enrollButton);
		//
		loginPanel.setBounds(0, 0, 272, 324);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		setBounds(dim.width / 2 - 291 / 2, dim.height / 2 - 372 / 2, 291, 372);
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
		usernameInput2 = new JTextField();
		usernameInput2.setBounds(410, 45, 150, 33);
		usernameInput2.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput2.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(usernameInput2);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(595, 110, 120, 23);
		enrollmentPanel.add(password);
		//
		passwordInput2 = new JTextField();
		passwordInput2.setBounds(410, 105, 150, 33);
		passwordInput2.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput2.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		enrollmentPanel.add(passwordInput2);
		//
		JLabel cityName = new JLabel("شهر");
		cityName.setFont(cityName.getFont().deriveFont(cityName.getFont().getSize() * 1.6F));
		cityName.setFont(cityName.getFont().deriveFont(Font.BOLD));
		cityName.setBounds(633, 170, 120, 23);
		enrollmentPanel.add(cityName);
		cityInput = new JTextField();
		cityInput.setBounds(410, 165, 150, 33);
		cityInput.setHorizontalAlignment(SwingConstants.RIGHT);
		cityInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		cityInput.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (e.getKeyChar() >= 48 && e.getKeyChar() <= 57) {
					String s = cityInput.getText();
					if (s.length() != 0) {
						cityInput.setText(s.substring(0, s.length() - 1));
					}
				}
			}
		});
		enrollmentPanel.add(cityInput);
		//
		JLabel firstName = new JLabel("نام");
		firstName.setFont(firstName.getFont().deriveFont(firstName.getFont().getSize() * 1.6F));
		firstName.setFont(firstName.getFont().deriveFont(Font.BOLD));
		firstName.setBounds(295, 50, 120, 23);
		enrollmentPanel.add(firstName);
		firstNameInput = new JTextField();
		firstNameInput.setBounds(30, 45, 150, 33);
		firstNameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		firstNameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		firstNameInput.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (e.getKeyChar() >= 48 && e.getKeyChar() <= 57) {
					String s = firstNameInput.getText();
					if (s.length() != 0) {
						firstNameInput.setText(s.substring(0, s.length() - 1));
					}
				}
			}
		});
		enrollmentPanel.add(firstNameInput);
		//
		JLabel lastName = new JLabel("نام خانوادگی");
		lastName.setFont(lastName.getFont().deriveFont(lastName.getFont().getSize() * 1.6F));
		lastName.setFont(lastName.getFont().deriveFont(Font.BOLD));
		lastName.setBounds(195, 110, 150, 23);
		enrollmentPanel.add(lastName);
		lastNameInput = new JTextField();
		lastNameInput.setBounds(30, 105, 150, 33);
		lastNameInput.setHorizontalAlignment(SwingConstants.RIGHT);
		lastNameInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		lastNameInput.addKeyListener(new KeyAdapter() {
			public void keyReleased(KeyEvent e) {
				if (e.getKeyChar() >= 48 && e.getKeyChar() <= 57) {
					String s = lastNameInput.getText();
					if (s.length() != 0) {
						lastNameInput.setText(s.substring(0, s.length() - 1));
					}
				}
			}
		});
		enrollmentPanel.add(lastNameInput);
		//
		JLabel nationalCode = new JLabel("کد ملی");
		nationalCode.setFont(nationalCode.getFont().deriveFont(nationalCode.getFont().getSize() * 1.6F));
		nationalCode.setFont(nationalCode.getFont().deriveFont(Font.BOLD));
		nationalCode.setBounds(248, 170, 120, 23);
		enrollmentPanel.add(nationalCode);
		nationalCodeInput = new JTextField();
		nationalCodeInput.setBounds(30, 165, 150, 33);
		nationalCodeInput.setHorizontalAlignment(SwingConstants.RIGHT);
		nationalCodeInput.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		nationalCodeInput.addKeyListener(new KeyAdapter() {
			public void keyReleased(KeyEvent e) {
				if (!(e.getKeyChar() >= 48 && e.getKeyChar() <= 57)) {
					String s = nationalCodeInput.getText();
					if (s.length() != 0) {
						nationalCodeInput.setText(s.substring(0, s.length() - 1));
					}
				}
			}
		});
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
		enrollButton2.addActionListener(this);
		enrollButton2.setActionCommand("Enrollment2");
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
		returnButton.addActionListener(this);
		returnButton.setActionCommand("Return");
		enrollmentPanel.add(returnButton);
		//
		//
		enrollmentPanel.setBounds(10, 10, 738, 363);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		setBounds(dim.width / 2 - 748 / 2, dim.height / 2 - 403 / 2, 748, 403);
		setSize(748, 403);
		//
		return enrollmentPanel;
		//
	}

	private JPanel getAdminPanel() {
		JPanel adminPanel = new JPanel();
		adminPanel.setBackground(Color.WHITE);
		GridBagLayout gbl = new GridBagLayout();
		gbl.columnWidths = new int[] { 100, 100, 100, 100, 100 };
		gbl.rowHeights = new int[] { 50, 50, 50, 50, 50, 50, 50, 50, 50 };
		gbl.columnWeights = new double[] { 1.0d, 1.0d, 1.0d, 1.0d, 1.0d };
		gbl.rowWeights = new double[] { 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d };
		adminPanel.setLayout(gbl);
		GridBagConstraints gbc1 = new GridBagConstraints(0, 0, 1, 1, 1.0d, 1.0d, GridBagConstraints.CENTER,
				GridBagConstraints.BOTH, new Insets(1, 1, 1, 1), 1, 1);
		GridBagConstraints gbc2 = new GridBagConstraints(1, 0, 1, 1, 1.0d, 1.0d, GridBagConstraints.CENTER,
				GridBagConstraints.BOTH, new Insets(1, 1, 1, 1), 1, 1);
		GridBagConstraints gbc3 = new GridBagConstraints(2, 0, 1, 1, 1.0d, 1.0d, GridBagConstraints.CENTER,
				GridBagConstraints.BOTH, new Insets(1, 1, 1, 1), 1, 1);
		GridBagConstraints gbc4 = new GridBagConstraints(3, 0, 1, 1, 1.0d, 1.0d, GridBagConstraints.CENTER,
				GridBagConstraints.BOTH, new Insets(1, 1, 1, 1), 1, 1);
		GridBagConstraints gbc5 = new GridBagConstraints(4, 0, 1, 1, 1.0d, 1.0d, GridBagConstraints.CENTER,
				GridBagConstraints.BOTH, new Insets(1, 1, 1, 1), 1, 1);

		returnButton2 = new JButton("خروج");
		employeeButton2 = new JButton("کارمندان");
		returnButton2.addActionListener(this);
		returnButton2.setActionCommand("Return2");
		employeeButton2.addActionListener(this);
		employeeButton2.setActionCommand("Employee");
		//
		JButton button3 = new JButton("-");
		JButton button4 = new JButton("-");
		JButton button5 = new JButton("-");
		adminPanel.add(returnButton2, gbc1);
		adminPanel.add(employeeButton2, gbc2);
		adminPanel.add(button3, gbc3);
		adminPanel.add(button4, gbc4);
		adminPanel.add(button5, gbc5);
		//
		return adminPanel;
	}

}
