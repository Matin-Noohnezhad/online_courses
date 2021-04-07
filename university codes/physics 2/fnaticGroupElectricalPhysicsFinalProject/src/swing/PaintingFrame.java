package swing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JFrame;
import javax.swing.JPanel;

import calculations.Calculate;

public class PaintingFrame extends JFrame {

	private static final long serialVersionUID = 5869377947025823101L;

	private JPanel PaintingPanel;
	// private double biggestElectricField = 1;

	public PaintingFrame() {
		setBounds(GUIManager.getChargesPanel().getLocationOnScreen().x - 10,
				GUIManager.getChargesPanel().getLocationOnScreen().y, GUIManager.getChargesPanel().getWidth() + 20,
				GUIManager.getChargesPanel().getHeight() + 10);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setResizable(false);
		//
		// for (int i = 0; i < GUIManager.getChargesPanel().getWidth() + 20;
		// i++) {
		// for (int j = 0; j < GUIManager.getChargesPanel().getHeight() + 10;
		// j++) {
		// if (Math.sqrt((Math
		// .pow(Calculate.electricFieldsX(GUIManager.getChargesPanel().getElectricCharges(),
		// i, j), 2)
		// +
		// Math.pow(Calculate.electricFieldsY(GUIManager.getChargesPanel().getElectricCharges(),
		// i, j),
		// 2))) > biggestElectricField) {
		// biggestElectricField = Math.sqrt((Math
		// .pow(Calculate.electricFieldsX(GUIManager.getChargesPanel().getElectricCharges(),
		// i, j), 2)
		// + Math.pow(
		// Calculate.electricFieldsY(GUIManager.getChargesPanel().getElectricCharges(),
		// i, j),
		// 2)));
		// }
		// }
		// }
		//
		PaintingPanel = new JPanel() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 8658144326527784997L;

			@Override
			protected void paintComponent(Graphics g) {
				super.paintComponent(g);
				Graphics2D g2 = (Graphics2D) g;
				for (int i = 0; i < GUIManager.getChargesPanel().getWidth() + 20; i++) {
					for (int j = 0; j < GUIManager.getChargesPanel().getHeight() + 10; j++) {
						double ElectricFieldMagnitude = Math.sqrt((Math.pow(
								Calculate.electricFieldsX(GUIManager.getChargesPanel().getElectricCharges(), i, j), 2)
								+ Math.pow(Calculate.electricFieldsY(GUIManager.getChargesPanel().getElectricCharges(),
										i, j), 2)));
						//
						if (ElectricFieldMagnitude / Math.pow(10, 8) > 9) {
							g2.setColor(new Color(0, 0, 0));
						} else if (ElectricFieldMagnitude / Math.pow(10, 8) > 7) {
							g2.setColor(new Color(5, 5, 5));
						} else if (ElectricFieldMagnitude / Math.pow(10, 8) > 5) {
							g2.setColor(new Color(10, 10, 10));
						} else if (ElectricFieldMagnitude / Math.pow(10, 8) > 3) {
							g2.setColor(new Color(15, 15, 15));
						} else if (ElectricFieldMagnitude / Math.pow(10, 8) > 1) {
							g2.setColor(new Color(21, 21, 21));
						} else if (ElectricFieldMagnitude / Math.pow(10, 7) > 9) {
							g2.setColor(new Color(26, 26, 26));
						} else if (ElectricFieldMagnitude / Math.pow(10, 7) > 7) {
							g2.setColor(new Color(31, 31, 31));
						} else if (ElectricFieldMagnitude / Math.pow(10, 7) > 5) {
							g2.setColor(new Color(36, 36, 36));
						} else if (ElectricFieldMagnitude / Math.pow(10, 7) > 3) {
							g2.setColor(new Color(41, 41, 41));
						} else if (ElectricFieldMagnitude / Math.pow(10, 7) > 1) {
							g2.setColor(new Color(47, 47, 47));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 9) {
							g2.setColor(new Color(52, 52, 52));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 8) {
							g2.setColor(new Color(54, 54, 54));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 7) {
							g2.setColor(new Color(57, 57, 57));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 6) {
							g2.setColor(new Color(60, 60, 60));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 5) {
							g2.setColor(new Color(62, 62, 62));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 4) {
							g2.setColor(new Color(65, 65, 65));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 3) {
							g2.setColor(new Color(67, 67, 67));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 2) {
							g2.setColor(new Color(70, 70, 70));
						} else if (ElectricFieldMagnitude / Math.pow(10, 6) > 1) {
							g2.setColor(new Color(73, 73, 73));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 9) {
							g2.setColor(new Color(78, 78, 78));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 8) {
							g2.setColor(new Color(80, 80, 80));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 7) {
							g2.setColor(new Color(83, 83, 83));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 6) {
							g2.setColor(new Color(86, 86, 86));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 5) {
							g2.setColor(new Color(88, 88, 88));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 4) {
							g2.setColor(new Color(90, 90, 90));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 3) {
							g2.setColor(new Color(93, 93, 93));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 2) {
							g2.setColor(new Color(96, 96, 96));
						} else if (ElectricFieldMagnitude / Math.pow(10, 5) > 1) {
							g2.setColor(new Color(99, 99, 99));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 9) {
							g2.setColor(new Color(104, 104, 104));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 8) {
							g2.setColor(new Color(106, 106, 106));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 7) {
							g2.setColor(new Color(109, 109, 109));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 6) {
							g2.setColor(new Color(112, 112, 112));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 5) {
							g2.setColor(new Color(114, 114, 114));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 4) {
							g2.setColor(new Color(116, 116, 116));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 3) {
							g2.setColor(new Color(119, 119, 119));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 2) {
							g2.setColor(new Color(122, 122, 122));
						} else if (ElectricFieldMagnitude / Math.pow(10, 4) > 1) {
							g2.setColor(new Color(125, 125, 125));
						} else if (ElectricFieldMagnitude / Math.pow(10, 3) > 9) {
							g2.setColor(new Color(130, 130, 130));
						} else if (ElectricFieldMagnitude / Math.pow(10, 3) > 7) {
							g2.setColor(new Color(135, 135, 135));
						} else if (ElectricFieldMagnitude / Math.pow(10, 3) > 5) {
							g2.setColor(new Color(140, 140, 140));
						} else if (ElectricFieldMagnitude / Math.pow(10, 3) > 3) {
							g2.setColor(new Color(145, 145, 145));
						} else if (ElectricFieldMagnitude / Math.pow(10, 3) > 1) {
							g2.setColor(new Color(150, 150, 150));
						} else if (ElectricFieldMagnitude / Math.pow(10, 2) > 7) {
							g2.setColor(new Color(154, 154, 154));
						} else if (ElectricFieldMagnitude / Math.pow(10, 2) > 5) {
							g2.setColor(new Color(159, 159, 159));
						} else if (ElectricFieldMagnitude / Math.pow(10, 2) > 3) {
							g2.setColor(new Color(164, 164, 164));
						} else if (ElectricFieldMagnitude / Math.pow(10, 2) > 1) {
							g2.setColor(new Color(170, 170, 170));
						} else if (ElectricFieldMagnitude / Math.pow(10, 1) > 8) {
							g2.setColor(new Color(175, 175, 175));
						} else if (ElectricFieldMagnitude / Math.pow(10, 1) > 6) {
							g2.setColor(new Color(195, 195, 195));
						} else if (ElectricFieldMagnitude / Math.pow(10, 1) > 4) {
							g2.setColor(new Color(215, 215, 215));
						} else if (ElectricFieldMagnitude / Math.pow(10, 1) > 2) {
							g2.setColor(new Color(235, 235, 235));
						} else if (ElectricFieldMagnitude / Math.pow(10, 1) > 0) {
							g2.setColor(new Color(255, 255, 255));
						}
						//
						g2.drawRect(i, j, 1, 1);
						//
					}
				}
			}

		};
		PaintingPanel.setBackground(Color.white);
		add(PaintingPanel);
		//

	}

	public JPanel getPaintingPanel() {
		return PaintingPanel;
	}

	public void setPaintingPanel(JPanel paintingPanel) {
		PaintingPanel = paintingPanel;
	}

}
