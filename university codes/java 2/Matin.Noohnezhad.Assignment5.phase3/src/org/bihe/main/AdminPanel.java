package org.bihe.main;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JPanel;

import org.bihe.bean.Position;

public class AdminPanel extends JPanel implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4570189581711622642L;

	public AdminPanel() {
		setBackground(Color.WHITE);
		GridBagLayout gbl = new GridBagLayout();
		gbl.columnWidths = new int[] { 100, 100, 100, 100, 100 };
		gbl.rowHeights = new int[] { 50, 50, 50, 50, 50, 50, 50, 50, 50 };
		gbl.columnWeights = new double[] { 1.0d, 1.0d, 1.0d, 1.0d, 1.0d };
		gbl.rowWeights = new double[] { 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d, 1.0d };
		setLayout(gbl);
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

		JButton returnButton2 = new JButton("خروج");
		JButton employeeButton2 = new JButton("کارمندان");
		returnButton2.addActionListener(this);
		returnButton2.setActionCommand("Return2");
		employeeButton2.addActionListener(this);
		employeeButton2.setActionCommand("Employee");
		//
		JButton button3 = new JButton("-");
		button3.setEnabled(false);
		JButton button4 = new JButton("-");
		button4.setEnabled(false);
		JButton button5 = new JButton("-");
		button5.setEnabled(false);
		add(returnButton2, gbc1);
		add(employeeButton2, gbc2);
		add(button3, gbc3);
		add(button4, gbc4);
		add(button5, gbc5);
		//
		setPreferredSize(new Dimension((int)Toolkit.getDefaultToolkit().getScreenSize().width,(int) Toolkit.getDefaultToolkit().getScreenSize().height));
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		switch (e.getActionCommand()) {
		case "Return2":
			// setSize(WIDTH, HEIGHT);
			GUIManager.getMainFrame().setTitle("لاگین شرکت پست آسمان");
//			GUIManager.getMainFrame().setContentPane(GUIManager.getLoginPanel());
			GUIManager.addLoginPanelToCenter();

			break;

		case "Employee":
			// setSize(WIDTH, HEIGHT);
			GUIManager.getMainFrame().setTitle("ثبت نام کارمند");
			GUIManager.addEnrollmentPanelToCenter();
//			GUIManager.getMainFrame().setContentPane(GUIManager.getEnrollmentPanel());
//			POSITION = Position.EMPLOYEE;
			GUIManager.getEnrollmentPanel().setPOSITION(Position.EMPLOYEE);
			
			break;

		default:

			break;
		}
	}

}
