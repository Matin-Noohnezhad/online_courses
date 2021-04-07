package swing;

import java.awt.Dimension;
import java.awt.GridLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class LeftPanelDimension extends JPanel {

	private static final long serialVersionUID = 1L;

	private JLabel label1 = new JLabel("__");
	private JLabel label2 = new JLabel("__");
	private JLabel label3 = new JLabel("__");
	private JLabel label4 = new JLabel("__");
	private JLabel label5 = new JLabel("__");
//	private JLabel label6 = new JLabel("__");
//	private JLabel label7 = new JLabel("__");
//	private JLabel label8 = new JLabel("__");

	public LeftPanelDimension() {

		setPreferredSize(new Dimension(20, 600));
		setLayout(new GridLayout(17, 1));
		add(new JLabel());
		add(new JLabel());
		add(label1);
		add(new JLabel());
		add(new JLabel());
		add(label2);
		add(new JLabel());
		add(new JLabel());
		add(label3);
		add(new JLabel());
		add(new JLabel());
		add(label4);
		add(new JLabel());
		add(new JLabel());
		add(label5);
		add(new JLabel());
		add(new JLabel());
//		add(label6);
//		add(label7);
//		add(label8);
	}

	public JLabel getLabel1() {
		return label1;
	}

	public void setLabel1(JLabel label1) {
		this.label1 = label1;
	}

	public JLabel getLabel2() {
		return label2;
	}

	public void setLabel2(JLabel label2) {
		this.label2 = label2;
	}

	public JLabel getLabel3() {
		return label3;
	}

	public void setLabel3(JLabel label3) {
		this.label3 = label3;
	}

	public JLabel getLabel4() {
		return label4;
	}

	public void setLabel4(JLabel label4) {
		this.label4 = label4;
	}

	public JLabel getLabel5() {
		return label5;
	}

	public void setLabel5(JLabel label5) {
		this.label5 = label5;
	}

//	public JLabel getLabel6() {
//		return label6;
//	}
//
//	public void setLabel6(JLabel label6) {
//		this.label6 = label6;
//	}
//
//	public JLabel getLabel7() {
//		return label7;
//	}
//
//	public void setLabel7(JLabel label7) {
//		this.label7 = label7;
//	}
//
//	public JLabel getLabel8() {
//		return label8;
//	}
//
//	public void setLabel8(JLabel label8) {
//		this.label8 = label8;
//	}
}