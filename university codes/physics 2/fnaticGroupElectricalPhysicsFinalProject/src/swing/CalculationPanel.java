package swing;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

public class CalculationPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	private boolean successfullDrag = true;
	private JTextField chargeValueTextField;
	private String ElectricFieldValue = "0";
	private String ElectricPotentialValue = "0";
	private JLabel electricPotentialValueLabel;
	private JLabel electricFieldValueLabel;
	private JPanel electricFieldDirection;
	private final int directionLineLength = 90;
	private double cosDirection = 1;
	private double sinDirection = 0;

	public CalculationPanel() {
		add(inputPanel());
		add(outputPanel());
	}

	private JPanel inputPanel() {
		JPanel northPanel = new JPanel(new FlowLayout());
		northPanel.setPreferredSize(new Dimension(290, 160));
		JLabel chargeValueLabel = new JLabel("Enter Value Of Charge Here:");
		chargeValueLabel.setFont(chargeValueLabel.getFont().deriveFont(Font.BOLD));
		chargeValueLabel.setFont(chargeValueLabel.getFont().deriveFont(chargeValueLabel.getFont().getSize() * 1.7f));
		chargeValueLabel.setPreferredSize(new Dimension(270, 80));
		//
		northPanel.add(chargeValueLabel);
		chargeValueTextField = new JTextField();
		chargeValueTextField.setText("1");
		chargeValueTextField.setPreferredSize(new Dimension(85, 40));
		chargeValueTextField
				.setFont(chargeValueTextField.getFont().deriveFont(chargeValueTextField.getFont().getSize() * 1.7f));
		chargeValueTextField.addKeyListener(new KeyListener() {

			@Override
			public void keyTyped(KeyEvent e) {

			}

			@Override
			public void keyReleased(KeyEvent e) {
				char c = e.getKeyChar();
				if (c == '1' || c == '2' || c == '3' || c == '4' || c == '5' || c == '6' || c == '7' || c == '8'
						|| c == '9' || c == '0') {
					if (Double.parseDouble(chargeValueTextField.getText()) > 1
							|| Double.parseDouble(chargeValueTextField.getText()) < -1)

					{
						JOptionPane.showMessageDialog(null, "Enter a number between 1 and -1");
						chargeValueTextField.setText("");
					}
				} else if (c == '-' || c == '.') {

				} else {
					chargeValueTextField.setText("");
				}
			}

			@Override
			public void keyPressed(KeyEvent e) {

			}
		});
		northPanel.add(chargeValueTextField);

		return northPanel;
	}

	private JPanel outputPanel() {
		JPanel centerPanel = new JPanel(new FlowLayout());
		centerPanel.setPreferredSize(new Dimension(290, 560));
		//
		JTextArea electricPotentialLabel = new JTextArea("ELectric Potential Value At\nThe Mouse Cursor Point:");
		electricPotentialLabel.setFont(electricPotentialLabel.getFont().deriveFont(Font.BOLD));
		electricPotentialLabel.setFont(
				electricPotentialLabel.getFont().deriveFont(electricPotentialLabel.getFont().getSize() * 1.5f));
		electricPotentialLabel.setEditable(false);
		electricPotentialLabel.setBackground(Color.lightGray);
		electricPotentialLabel.setPreferredSize(new Dimension(270, 50));
		//
		electricPotentialValueLabel = new JLabel(ElectricPotentialValue);
		electricPotentialValueLabel.setFont(electricPotentialValueLabel.getFont().deriveFont(Font.BOLD));
		electricPotentialValueLabel.setFont(electricPotentialValueLabel.getFont()
				.deriveFont(electricPotentialValueLabel.getFont().getSize() * 1.5f));
		electricPotentialValueLabel.setForeground(Color.blue);
		electricPotentialValueLabel.setPreferredSize(new Dimension(270, 40));
		centerPanel.add(electricPotentialLabel);
		centerPanel.add(electricPotentialValueLabel);
		//
		JTextArea electricFieldLabel = new JTextArea("ELectric Field Value At The\nMouse Cursor Point:");
		electricFieldLabel.setFont(electricFieldLabel.getFont().deriveFont(Font.BOLD));
		electricFieldLabel
				.setFont(electricFieldLabel.getFont().deriveFont(electricFieldLabel.getFont().getSize() * 1.5f));
		electricFieldLabel.setEditable(false);
		electricFieldLabel.setBackground(Color.lightGray);
		electricFieldLabel.setPreferredSize(new Dimension(270, 50));
		//
		electricFieldValueLabel = new JLabel(ElectricFieldValue);
		electricFieldValueLabel.setFont(electricFieldValueLabel.getFont().deriveFont(Font.BOLD));
		electricFieldValueLabel.setFont(
				electricFieldValueLabel.getFont().deriveFont(electricFieldValueLabel.getFont().getSize() * 1.5f));
		electricFieldValueLabel.setForeground(Color.blue);
		electricFieldValueLabel.setPreferredSize(new Dimension(270, 40));
		centerPanel.add(electricFieldLabel);
		centerPanel.add(electricFieldValueLabel);
		//
		JLabel electricFieldDirectionLabel = new JLabel("Electric Field Direction:");
		electricFieldDirectionLabel.setFont(electricFieldDirectionLabel.getFont().deriveFont(Font.BOLD));
		electricFieldDirectionLabel.setFont(electricFieldDirectionLabel.getFont()
				.deriveFont(electricFieldDirectionLabel.getFont().getSize() * 1.5f));
		centerPanel.add(electricFieldDirectionLabel);
		//
		electricFieldDirection = new JPanel() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 4592785544860615352L;

			@Override
			protected void paintComponent(Graphics g) {
				super.paintComponent(g);
				Graphics2D g2 = (Graphics2D) g;
				//
				g2.setColor(Color.blue);
				g2.setStroke(new BasicStroke(3));
				g2.drawLine(this.getWidth() / 2, this.getHeight() / 2,
						(this.getWidth() / 2) + (int) (directionLineLength * cosDirection),
						(this.getHeight() / 2) + (int) (directionLineLength * sinDirection));
				//
				g2.setColor(Color.black);
				g2.fillOval((this.getWidth() / 2) - 10, (this.getHeight() / 2) - 10, 20, 20);
				//
			}
		};
		electricFieldDirection.setPreferredSize(new Dimension(270, 200));
		electricFieldDirection.setBackground(Color.orange);
		centerPanel.add(electricFieldDirection);
		//
		JButton paintings = new JButton("Paintings(Electric Field)");
		paintings.setFont(paintings.getFont().deriveFont(Font.BOLD));
		paintings.setFont(paintings.getFont().deriveFont(paintings.getFont().getSize() * 1.3f));
		paintings.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				new PaintingFrame().setVisible(true);
			}
		});
		JButton potential = new JButton("Equipotential Lines");
		potential.setFont(potential.getFont().deriveFont(Font.BOLD));
		potential.setFont(potential.getFont().deriveFont(potential.getFont().getSize() * 1.3f));
		potential.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				new EquipotentialLinesFrame().setVisible(true);
			}
		});
		//
		centerPanel.add(paintings);
		centerPanel.add(potential);
		//
		centerPanel.setBackground(Color.lightGray);
		//
		return centerPanel;
	}

	public JTextField getChargeValueTextField() {
		return chargeValueTextField;
	}

	public void setChargeValueTextField(JTextField chargeValueTextField) {
		this.chargeValueTextField = chargeValueTextField;
	}

	public boolean isSuccessfullDrag() {
		return successfullDrag;
	}

	public void setSuccessfullDrag(boolean successfullDrag) {
		this.successfullDrag = successfullDrag;
	}

	public String getElectricPotentialValue() {
		return ElectricPotentialValue;
	}

	public void setElectricPotentialValue(String electricPotentialValue) {
		this.ElectricPotentialValue = electricPotentialValue;
		electricPotentialValueLabel.setText(this.ElectricPotentialValue);
	}

	public String getElectricFieldValue() {
		return ElectricFieldValue;
	}

	public void setElectricFieldValue(String electricFieldValue) {
		this.ElectricFieldValue = electricFieldValue;
		electricFieldValueLabel.setText(this.ElectricFieldValue);
	}

	public double getCosDirection() {
		return cosDirection;
	}

	public void setCosDirection(double cosDirection) {
		this.cosDirection = cosDirection;
	}

	public double getSinDirection() {
		return sinDirection;
	}

	public void setSinDirection(double sinDirection) {
		this.sinDirection = sinDirection;
	}
}