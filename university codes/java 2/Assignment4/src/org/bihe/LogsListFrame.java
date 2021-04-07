package org.bihe;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;

public class LogsListFrame extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private JPanel contentPane;

	/**
	 * Create the frame.
	 */
	public LogsListFrame(ArrayList<Log> readLogs) {
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setSize(400, 600);
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		this.setLocation(dim.width / 2 - this.getSize().width / 2, dim.height / 2 - this.getSize().height / 2);
		setResizable(false);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(34, 34, 329, 479);
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
		tableModel.addColumn("Method Call");
		tableModel.addColumn("Date");

		for (int i = 0; i < readLogs.size(); i++) {
			SimpleDateFormat dt1 = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");

			String date = dt1.format(readLogs.get(i).getStartActionDate());

			Object[] row = { new Integer(i + 1), String.valueOf(readLogs.get(i).getMethodCall()).replaceAll("_", " "),
					date };
			tableModel.addRow(row);
		}
		JTable table = new JTable();
		table.setModel(tableModel);

		table.getColumn("No").setMaxWidth(50);

		scrollPane.setViewportView(table);

		JButton btnShow = new JButton("Show");
		btnShow.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				int selected = table.getSelectedRow();
				if (selected != -1) {
					ShowLogFrame slf = new ShowLogFrame(readLogs.get(selected));
					slf.setVisible(true);
				}
			}
		});
		btnShow.setBounds(152, 537, 89, 23);
		contentPane.add(btnShow);

	}
}
