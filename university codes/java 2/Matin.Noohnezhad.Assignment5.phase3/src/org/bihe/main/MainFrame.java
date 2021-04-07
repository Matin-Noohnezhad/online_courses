package org.bihe.main;

import java.awt.BorderLayout;
import java.awt.Toolkit;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class MainFrame extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = -80429907613085374L;
//	private static final int WIDTH = 800;
//	private static final int HEIGHT = 500;
	private JPanel mainPanel;

	public MainFrame() {
//		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
//		setBounds(dim.width / 2 - WIDTH / 2, dim.height / 2 - HEIGHT / 2, WIDTH, HEIGHT);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		//
		setIconImage(Resource.getImage("pic3.png"));
		//
		setLayout(new BorderLayout());
		//
		this.setTitle("لاگین شرکت پست آسمان");
		//
		mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		add(mainPanel,BorderLayout.CENTER);
		// add(getLoginPanel(),BorderLayout.CENTER);
		// pack();
	}

	public void addToCenter(JPanel panel) {
		mainPanel.removeAll();
		mainPanel.add(panel, BorderLayout.CENTER);
		mainPanel.repaint();
		mainPanel.revalidate();
		pack();
		int wid = getWidth();
		int hei = getHeight();
		setBounds((int) Toolkit.getDefaultToolkit().getScreenSize().getWidth() / 2 - wid / 2,
				(int) Toolkit.getDefaultToolkit().getScreenSize().getHeight() / 2 - hei / 2, wid, hei);
	}

}
