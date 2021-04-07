package swing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JFrame;
import javax.swing.JPanel;

import calculations.Calculate;

public class EquipotentialLinesFrame extends JFrame {

	private static final long serialVersionUID = 5944304377231162452L;

	private JPanel EquipotentialPanel;

	public EquipotentialLinesFrame() {
		setBounds(GUIManager.getChargesPanel().getLocationOnScreen().x - 10,
				GUIManager.getChargesPanel().getLocationOnScreen().y, GUIManager.getChargesPanel().getWidth() + 20,
				GUIManager.getChargesPanel().getHeight() + 10);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setResizable(false);
		//

		EquipotentialPanel = new JPanel() {
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
						double electricPotencial = Math.abs(
								Calculate.electricPotential(GUIManager.getChargesPanel().getElectricCharges(), i, j));

						if (electricPotencial > 10 && electricPotencial < 10.2)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 20 && electricPotencial < 20.3)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 30 && electricPotencial < 30.4)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 40 && electricPotencial < 40.5)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 50 && electricPotencial < 50.6)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 60 && electricPotencial < 61)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 70 && electricPotencial < 72)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 80 && electricPotencial < 83)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 90 && electricPotencial < 92)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 145 && electricPotencial < 147)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 245 && electricPotencial < 247)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 345 && electricPotencial < 349)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 455 && electricPotencial < 458)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 590 && electricPotencial < 595)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 700 && electricPotencial < 705)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 800 && electricPotencial < 805)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 900 && electricPotencial < 905)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1000 && electricPotencial < 1005)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1150 && electricPotencial < 1156)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1350 && electricPotencial < 1357)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1600 && electricPotencial < 1610)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1850 && electricPotencial < 1860)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2200 && electricPotencial < 2220)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2500 && electricPotencial < 2520)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2800 && electricPotencial < 2830)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3050 && electricPotencial < 3090)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3400 && electricPotencial < 3450)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3800 && electricPotencial < 3850)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 4500 && electricPotencial < 4560)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 5000 && electricPotencial < 5080)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 5900 && electricPotencial < 6000)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 6800 && electricPotencial < 6900)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 8000 && electricPotencial < 8100)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 10000 && electricPotencial < 10100)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 12000 && electricPotencial < 12100)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 14000 && electricPotencial < 14100)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 15500 && electricPotencial < 15600)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 18000 && electricPotencial < 18200)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 21000 && electricPotencial < 21200)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 25000 && electricPotencial < 25200)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 30000 && electricPotencial < 30300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 35000 && electricPotencial < 35300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 42000 && electricPotencial < 42300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 50000 && electricPotencial < 50300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 60000 && electricPotencial < 60300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 70000 && electricPotencial < 70300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 80000 && electricPotencial < 80300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 90000 && electricPotencial < 90300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 100000 && electricPotencial < 100400)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 120000 && electricPotencial < 120400)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 150000 && electricPotencial < 150500)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 190000 && electricPotencial < 190500)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 250000 && electricPotencial < 250600)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 320000 && electricPotencial < 320600)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 370000 && electricPotencial < 370700)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 450000 && electricPotencial < 450700)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 580000 && electricPotencial < 580800)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 700000 && electricPotencial < 701000)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 850000 && electricPotencial < 851000)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1000000 && electricPotencial < 1001300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1200000 && electricPotencial < 1202000)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1300000 && electricPotencial < 1302000)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1500000 && electricPotencial < 1502300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1600000 && electricPotencial < 1602300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1700000 && electricPotencial < 1704300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1800000 && electricPotencial < 1805300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 1900000 && electricPotencial < 1906300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2100000 && electricPotencial < 2107300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2400000 && electricPotencial < 2408300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 2700000 && electricPotencial < 2709300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3000000 && electricPotencial < 3010300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3300000 && electricPotencial < 3311300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 3700000 && electricPotencial < 3711200)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 4100000 && electricPotencial < 4112300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 4700000 && electricPotencial < 4722300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 5300000 && electricPotencial < 5322300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 5900000 && electricPotencial < 5932300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 6500000 && electricPotencial < 6542300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 7200000 && electricPotencial < 7252300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 7900000 && electricPotencial < 7962300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 8600000 && electricPotencial < 8672300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 9500000 && electricPotencial < 9582300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 11000000 && electricPotencial < 11092300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 12000000 && electricPotencial < 12092300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 13000000 && electricPotencial < 13092300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 14000000 && electricPotencial < 14092300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 15000000 && electricPotencial < 15092300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 16000000 && electricPotencial < 16132300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 17000000 && electricPotencial < 17142300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 18000000 && electricPotencial < 18152300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 19000000 && electricPotencial < 19162300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 22000000 && electricPotencial < 22172300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 24000000 && electricPotencial < 24182300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 26000000 && electricPotencial < 26202300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 28000000 && electricPotencial < 28202300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 33000000 && electricPotencial < 33202300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 36000000 && electricPotencial < 36242300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 39000000 && electricPotencial < 39852300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 44000000 && electricPotencial < 44852300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 50000000 && electricPotencial < 50952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 55000000 && electricPotencial < 55952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 60000000 && electricPotencial < 60952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 66000000 && electricPotencial < 66952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 71000000 && electricPotencial < 71952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 78000000 && electricPotencial < 78952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 85000000 && electricPotencial < 85952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 95000000 && electricPotencial < 95952300)
							g2.setColor(Color.BLACK);
						else if (electricPotencial > 100000000 && electricPotencial < 100952300)
							g2.setColor(Color.BLACK);

						g2.drawRect(i, j, 1, 1);

						g2.setColor(Color.WHITE);

					}
				}
			}

		};
		EquipotentialPanel.setBackground(Color.white);
		add(EquipotentialPanel);
		//

	}

	public JPanel getEquipotentialPanel() {
		return EquipotentialPanel;
	}

	public void setEquipotentialPanel(JPanel EquipotentialPanel) {
		this.EquipotentialPanel = EquipotentialPanel;
	}

}
