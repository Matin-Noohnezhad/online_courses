package swing;

import java.awt.Dimension;
import java.awt.Toolkit;

import javax.swing.JFrame;
import javax.swing.JSplitPane;

public class Frame extends JFrame {

	private static final long serialVersionUID = 1300373758042648457L;

	Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
	private final int WIDTH = 2 * dim.width / 3;
	private final int HEIGHT = 2 * dim.height / 3;
	int x = dim.width / 2 - WIDTH / 2;
	int y = dim.height / 2 - HEIGHT / 2;

	public Frame() {

		setTitle("Frame");
		setLocation(x, y);
		setSize(WIDTH, HEIGHT);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		// split two parts of page
		JSplitPane splitPane = new JSplitPane();
		splitPane.setDividerLocation(3 * WIDTH / 4);
		splitPane.setEnabled(false);
		splitPane.setLeftComponent(GUIManager.getShowPanel());
		splitPane.setRightComponent(GUIManager.getCalculationPanel());
		setResizable(false);
		add(splitPane);
	}

}