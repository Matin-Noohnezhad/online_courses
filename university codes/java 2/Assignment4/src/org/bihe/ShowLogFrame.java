package org.bihe;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;

public class ShowLogFrame extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private JPanel contentPane;

	/**
	 * Create the frame.
	 */
	public ShowLogFrame(Log log) {
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setSize(550, 300);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		this.setLocation(dim.width / 2 - this.getSize().width / 2, dim.height / 2 - this.getSize().height / 2);
		setResizable(false);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		SimpleDateFormat dt1 = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");

		String labelText = log.getMethodCall().toString().replace("_", " ") + " is started in "
				+ dt1.format(log.getStartActionDate()) + " and the duration was " + log.getTimeOfAction() / 1000
				+ " seconds";
		JLabel lblNewLabel = new JLabel(labelText);
		lblNewLabel.setFont(new Font("Tahoma", Font.PLAIN, 12));
		lblNewLabel.setBounds(20, 32, 495, 44);
		contentPane.add(lblNewLabel);

		JScrollPane scrollPane = new JScrollPane();

		scrollPane.setBounds(55, 114, 429, 127);
		contentPane.add(scrollPane);

		DefaultTableModel tableModel = new DefaultTableModel() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int column) {
				// all cells false
				return false;
			}
		};

		tableModel.addColumn("No");
		tableModel.addColumn("Info");

		ArrayList<String> informations = log.getMoreInfo();
		for (int i = 0; i < informations.size(); i++) {
			Object[] row = { new Integer(i + 1), informations.get(i) };
			tableModel.addRow(row);
		}
		JTable table = new JTable();
		table.setModel(tableModel);

		table.getColumn("No").setMaxWidth(100);

		scrollPane.setViewportView(table);

	}
}
