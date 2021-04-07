package swing;

import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.JLabel;
import javax.swing.JPanel;

public class DownPanelDimension extends JPanel {

	private static final long serialVersionUID = 1L;

	private JLabel label1 = new JLabel("cm");
	private JLabel label2 = new JLabel("10");
	private JLabel label3 = new JLabel("|");
	private JLabel label4 = new JLabel("|");
	private JLabel label5 = new JLabel("|");
	private JLabel label6 = new JLabel("|");
	private JLabel label7 = new JLabel("|");
	private JLabel label8 = new JLabel("|");

	public DownPanelDimension() {
		setPreferredSize(new Dimension(800, 20));
		setLayout(new GridLayout(1, 20));
		add(label1);
		add(label2);
		add(label3);
		add(label4);
		add(label5);
		add(label6);
		add(label7);
		add(label8);
	}

	public JLabel getLabel2() {
		return label2;
	}

	public void setLabel2(JLabel label2) {
		this.label2 = label2;
	}

}