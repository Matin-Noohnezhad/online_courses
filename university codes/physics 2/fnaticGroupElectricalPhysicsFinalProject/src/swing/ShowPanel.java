package swing;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Toolkit;

import javax.swing.JPanel;

public class ShowPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	public ShowPanel() {

		setBackground(Color.WHITE);

		setBounds(Toolkit.getDefaultToolkit().getScreenSize().width / 6,
				Toolkit.getDefaultToolkit().getScreenSize().height / 6,
				3 * Toolkit.getDefaultToolkit().getScreenSize().width / 6,
				4 * Toolkit.getDefaultToolkit().getScreenSize().height / 6);
		setLayout(new BorderLayout());
		add(GUIManager.getChargesPanel(), BorderLayout.CENTER);
		add(GUIManager.getDownPanelDimension(), BorderLayout.SOUTH);
		add(GUIManager.getLeftPanelDimension(), BorderLayout.WEST);
	}
}