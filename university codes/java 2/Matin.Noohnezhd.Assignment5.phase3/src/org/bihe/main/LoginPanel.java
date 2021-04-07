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
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import org.bihe.DAO.UserPassDAO;
import org.bihe.bean.Position;

public class LoginPanel extends JPanel implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5137362691099248534L;
	private JButton enrollButton = new JButton("ثبت نام");
	private JButton entryButton = new JButton("ورود");
	private JPasswordField passwordInput1;
	private static int enr = 0;
	private static int ent = 0;
	private JTextField usernameInput1;

	public LoginPanel() {

		setBorder(BorderFactory.createLineBorder(Color.BLACK, 7));
		setBackground(Color.WHITE);
		setLayout(null);
		//
		JLabel postOfficeName = new JLabel("پست آسمان");
		postOfficeName.setFont(postOfficeName.getFont().deriveFont(postOfficeName.getFont().getSize() * 1.6F));
		postOfficeName.setFont(postOfficeName.getFont().deriveFont(Font.BOLD));
		// postOfficeName.setBounds(75, 10, 180, 23);
		postOfficeName.setPreferredSize(new Dimension(180, 23));
		add(postOfficeName);
		//
		JLabel username = new JLabel("نام کاربری");
		username.setFont(username.getFont().deriveFont(username.getFont().getSize() * 1.6F));
		username.setFont(username.getFont().deriveFont(Font.BOLD));
		username.setBounds(155, 65, 135, 23);
		add(username);
		usernameInput1 = new JTextField();
		usernameInput1.setBounds(10, 60, 135, 33);
		usernameInput1.setHorizontalAlignment(SwingConstants.RIGHT);
		usernameInput1.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		usernameInput1.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (String.valueOf(passwordInput1.getPassword()).length() != 0 && usernameInput1.getText().length() != 0) {
					entryButton.setEnabled(true);
				} else {
					entryButton.setEnabled(false);
				}
			}
		});
		add(usernameInput1);
		//
		JLabel password = new JLabel("رمز عبور");
		password.setFont(password.getFont().deriveFont(password.getFont().getSize() * 1.6F));
		password.setFont(password.getFont().deriveFont(Font.BOLD));
		password.setBounds(165, 115, 135, 23);
		add(password);
		passwordInput1 = new JPasswordField();
		passwordInput1.setBounds(10, 110, 135, 33);
		passwordInput1.setHorizontalAlignment(SwingConstants.RIGHT);
		passwordInput1.setBorder(BorderFactory.createLineBorder(Color.BLACK, 5));
		passwordInput1.addKeyListener(new KeyAdapter() {
			@Override
			public void keyReleased(KeyEvent e) {
				if (usernameInput1.getText().length() != 0 && String.valueOf(passwordInput1.getPassword()).length() != 0) {
					entryButton.setEnabled(true);
				} else {
					entryButton.setEnabled(false);
				}
			}
		});
		add(passwordInput1);
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
		add(entryButton);
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
		add(enrollButton);
		//
		// setBounds(0, 0, 272, 324);
		setPreferredSize(new Dimension(272, 324));
		// Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		// setBounds(dim.width / 2 - 291 / 2, dim.height / 2 - 372 / 2, 291,
		// 372);
		//

	}

	@Override
	public void actionPerformed(ActionEvent e) {
		switch (e.getActionCommand()) {
		case "Enrollment":
			// setSize(WIDTH, HEIGHT);
			GUIManager.getMainFrame().setTitle("ثبت نام کاربر");
			// ????
			GUIManager.addEnrollmentPanelToCenter();
//			GUIManager.getMainFrame().setContentPane(GUIManager.getEnrollmentPanel());
			// ????
			// POSITION = Position.USER;
			GUIManager.getEnrollmentPanel().setPOSITION(Position.USER);

			passwordInput1.setText("");
			usernameInput1.setText("");
			break;

		case "Entry":
			// JOptionPane.showMessageDialog(this, "به صفحه مورد نطر وارد
			// شدید");
			String username2 = usernameInput1.getText();
			String password2 = String.valueOf(passwordInput1.getPassword());
			UserPassDAO upd2 = UserPassDAO.getInstance();
			Position position2 = upd2.checkUserPass(username2, password2);
			if (position2 == null) {
				JOptionPane.showMessageDialog(this, "چنین نام کاربری و رمز عبوری در سایت وجود ندارد.", "خطا",
						JOptionPane.INFORMATION_MESSAGE);
			} else if (position2 == Position.ADMIN) {
				GUIManager.getMainFrame().setExtendedState(JFrame.MAXIMIZED_BOTH);
				GUIManager.getMainFrame().setTitle("سیستم مرسولات پستی آسمان");
//				GUIManager.getMainFrame().getContentPane().removeAll();
				//
				// setLayout(new BorderLayout());
				// add(getAdminPanel(), BorderLayout.CENTER);
				GUIManager.addAdminPanelToCenter();
				//
				JOptionPane.showMessageDialog(this, position2 + " was entered.");
			} else {

				GUIManager.getMainFrame().setExtendedState(JFrame.MAXIMIZED_BOTH);
				GUIManager.getMainFrame().setTitle("سیستم مرسولات پستی آسمان");
				GUIManager.getMainFrame().getContentPane().removeAll();
				GUIManager.getMainFrame().getContentPane().setBackground(Color.cyan);
				JOptionPane.showMessageDialog(this, position2 + " was entered.");
			}
			passwordInput1.setText("");
			usernameInput1.setText("");
			break;
		default:
			break;
		}

	}
}
