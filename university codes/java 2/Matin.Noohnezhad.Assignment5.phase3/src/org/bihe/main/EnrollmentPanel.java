package org.bihe.main;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import org.bihe.DAO.UserPassDAO;
import org.bihe.bean.Position;
import org.bihe.bean.User;

public class EnrollmentPanel extends JPanel implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3811128194862047098L;
	private JButton returnButton = new JButton("بازگشت");
	private JButton enrollButton2 = new JButton("ثبت نام");
	private static int ret = 0;
	private static int enr2 = 0;
	private Position POSITION = Position.USER;
	private JTextField usernameInput2;
	private JPasswordField passwordInput2;
	private JTextField nationalCodeInput;
	private JTextField lastNameInput;
	private JTextField firstNameInput;
	private JTextField cityInput;

	public EnrollmentPanel() {
		setBorder(BorderFactory.createLineBorder(Color.BLACK, 6));
		setBackground(Color.WHITE);
		setLayout(null);
		//
		JLabel username = new JLabel("نام کاربری");
		username.setFont(username.getFont().deriveFont(username.getFont().getSize() * 1.6F));
		username.setFont(username.getFont().deriveFont(Font.BOLD));
		username.setBounds(580, 50, 120, 23);
		add(username);
		usernameInput2 = new JTextField();
		usernameInput2.setBounds(410, 45, 150, 33);
		usernameInput2.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput2.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		add(usernameInput2);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(595, 110, 120, 23);
		add(password);
		passwordInput2 = new JPasswordField();
		passwordInput2.setBounds(410, 105, 150, 33);
		passwordInput2.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput2.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		add(passwordInput2);
		//
		JLabel cityName = new JLabel("شهر");
		cityName.setFont(cityName.getFont().deriveFont(cityName.getFont().getSize() * 1.6F));
		cityName.setFont(cityName.getFont().deriveFont(Font.BOLD));
		cityName.setBounds(633, 170, 120, 23);
		add(cityName);
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
		add(cityInput);
		//
		JLabel firstName = new JLabel("نام");
		firstName.setFont(firstName.getFont().deriveFont(firstName.getFont().getSize() * 1.6F));
		firstName.setFont(firstName.getFont().deriveFont(Font.BOLD));
		firstName.setBounds(295, 50, 120, 23);
		add(firstName);
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
		add(firstNameInput);
		//
		JLabel lastName = new JLabel("نام خانوادگی");
		lastName.setFont(lastName.getFont().deriveFont(lastName.getFont().getSize() * 1.6F));
		lastName.setFont(lastName.getFont().deriveFont(Font.BOLD));
		lastName.setBounds(195, 110, 150, 23);
		add(lastName);
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
		add(lastNameInput);
		//
		JLabel nationalCode = new JLabel("کد ملی");
		nationalCode.setFont(nationalCode.getFont().deriveFont(nationalCode.getFont().getSize() * 1.6F));
		nationalCode.setFont(nationalCode.getFont().deriveFont(Font.BOLD));
		nationalCode.setBounds(248, 170, 120, 23);
		add(nationalCode);
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
		add(nationalCodeInput);
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
		add(enrollButton2);
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
		add(returnButton);
		//
		//
		setPreferredSize(new Dimension(738, 363));
		// setBounds(10, 10, 738, 363);
		// Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		// setBounds(dim.width / 2 - 748 / 2, dim.height / 2 - 403 / 2, 748,
		// 403);
		// setSize(748, 403);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		switch (e.getActionCommand()) {
		case "Return":

			// setSize(WIDTH, HEIGHT);
			GUIManager.getMainFrame().setTitle("لاگین شرکت پست آسمان");
			GUIManager.addLoginPanelToCenter();
			// GUIManager.getMainFrame().setContentPane(GUIManager.getLoginPanel());
			// GUIManager.addLoginPanelToCenter();
			break;
		case "Enrollment2":
			// JOptionPane.showMessageDialog(this, "شما با موفقیت ثبت نام شدین
			// :)");
			if (usernameInput2.getText().length() == 0 && String.valueOf(passwordInput2.getPassword()).length() == 0
					&& cityInput.getText().length() == 0 && firstNameInput.getText().length() == 0
					&& lastNameInput.getText().length() == 0 && nationalCodeInput.getText().length() == 0) {
				JOptionPane.showMessageDialog(this, "همه فیلدها باید پر باشد.", "خطا", JOptionPane.INFORMATION_MESSAGE);
			} else if (nationalCodeInput.getText().length() > 10) {
				JOptionPane.showMessageDialog(this, "تعداد ارقام کد ملی باید کمتر از 10 عدد باشد.", "خطا",
						JOptionPane.INFORMATION_MESSAGE);
			} else {
				String username = usernameInput2.getText();
				String password = String.valueOf(passwordInput2.getPassword());
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
		default:
			break;
		}
	}

	public Position getPOSITION() {
		return POSITION;
	}

	public void setPOSITION(Position pOSITION) {
		POSITION = pOSITION;
	}

}
