package swing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.util.ArrayList;

import javax.swing.JPanel;

import bean.ElectricCharge;
import calculations.Calculate;

public class ChargesPanel extends JPanel implements MouseWheelListener, MouseListener, MouseMotionListener {

	private static final long serialVersionUID = 1L;

	private int mouseX;
	private int mouseY;
	private ArrayList<ElectricCharge> electricCharges;

	public ChargesPanel() {

		electricCharges = new ArrayList<>();
		setBackground(Color.WHITE);
		addMouseWheelListener(this);
		addMouseListener(this);
		addMouseMotionListener(this);
	}

	public void mouseWheelMoved(MouseWheelEvent e) {
		if (e.getWheelRotation() == 1) {
			double magnitude = Double.parseDouble(GUIManager.getDownPanelDimension().getLabel2().getText()) / 10;
			if (magnitude > 0.00001) {
				GUIManager.getDownPanelDimension().getLabel2().setText(String.valueOf(magnitude));
			}

		} else if (e.getWheelRotation() == -1) {
			double magnitude = Double.parseDouble(GUIManager.getDownPanelDimension().getLabel2().getText()) * 10;
			if (magnitude < 100000) {
				GUIManager.getDownPanelDimension().getLabel2().setText(String.valueOf(magnitude));
			}
		}
	}

	public void mouseClicked(MouseEvent e) {
		ElectricCharge electricCharge = new ElectricCharge(mouseX, mouseY,
				Double.parseDouble(GUIManager.getCalculationPanel().getChargeValueTextField().getText()));
		electricCharges.add(electricCharge);
		this.repaint();
	}

	public void mousePressed(MouseEvent e) {
	}

	public void mouseReleased(MouseEvent e) {
	}

	public void mouseEntered(MouseEvent e) {
	}

	public void mouseExited(MouseEvent e) {
	}

	public void mouseDragged(MouseEvent e) {
	}

	public void mouseMoved(MouseEvent e) {
		mouseX = e.getX();
		mouseY = e.getY();
		if (electricCharges.size() != 0) {
			//
			GUIManager.getCalculationPanel()
					.setElectricPotentialValue(String.valueOf((Calculate.electricPotential(electricCharges, mouseX,
							mouseY)) * GUIManager.getChargesPanel().getWidth() / (8
									* Double.parseDouble(GUIManager.getDownPanelDimension().getLabel2().getText()))));
			//
			GUIManager.getCalculationPanel()
					.setElectricFieldValue(String
							.valueOf((Math.sqrt((Math.pow(Calculate.electricFieldsX(electricCharges, mouseX, mouseY), 2)
									+ Math.pow(Calculate.electricFieldsY(electricCharges, mouseX, mouseY), 2))))
									* Math.pow(GUIManager.getChargesPanel().getWidth(), 2)
									/ Math.pow(
											(8 * Double.parseDouble(
													GUIManager.getDownPanelDimension().getLabel2().getText()) / 100),
											2)));
			//
			GUIManager.getCalculationPanel()
					.setCosDirection(Calculate.electricFieldsX(electricCharges, mouseX, mouseY)
							/ Math.sqrt((Math.pow(Calculate.electricFieldsX(electricCharges, mouseX, mouseY), 2)
									+ Math.pow(Calculate.electricFieldsY(electricCharges, mouseX, mouseY), 2))));
			GUIManager.getCalculationPanel()
					.setSinDirection(Calculate.electricFieldsY(electricCharges, mouseX, mouseY)
							/ Math.sqrt((Math.pow(Calculate.electricFieldsX(electricCharges, mouseX, mouseY), 2)
									+ Math.pow(Calculate.electricFieldsY(electricCharges, mouseX, mouseY), 2))));
			//
			GUIManager.getCalculationPanel().repaint();
			//
		}
		this.repaint();
	}

	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D) g;
		if (GUIManager.getCalculationPanel().isSuccessfullDrag()) {
			g2.setColor(Color.cyan);
			g2.fillOval(mouseX - 15, mouseY - 15, 30, 30);
		}
		for (int i = 0; i < electricCharges.size(); i++) {
			g2.setColor(Color.PINK);
			g2.fillOval(electricCharges.get(i).getX() - 15, electricCharges.get(i).getY() - 15, 30, 30);
		}
	}

	public int getMouseX() {
		return mouseX;
	}

	public void setMouseX(int mouseX) {
		this.mouseX = mouseX;
	}

	public int getMouseY() {
		return mouseY;
	}

	public void setMouseY(int mouseY) {
		this.mouseY = mouseY;
	}

	public ArrayList<ElectricCharge> getElectricCharges() {
		return electricCharges;
	}

	public void setElectricCharges(ArrayList<ElectricCharge> electricCharges) {
		this.electricCharges = electricCharges;
	}
}