package swing;

public class GUIManager {

	private static ShowPanel showPanel;
	private static ChargesPanel chargesPanel;
	private static DownPanelDimension downPanelDimension;
	private static LeftPanelDimension leftPanelDimension;
	private static CalculationPanel calculationPanel;

	public static ChargesPanel getChargesPanel() {
		if (chargesPanel == null) {
			chargesPanel = new ChargesPanel();
		}
		return chargesPanel;
	}

	public static void setChargesPanel(ChargesPanel chargesPanel) {
		GUIManager.chargesPanel = chargesPanel;
	}

	public static DownPanelDimension getDownPanelDimension() {
		if (downPanelDimension == null) {
			downPanelDimension = new DownPanelDimension();
		}
		return downPanelDimension;
	}

	public static void setDownPanelDimension(DownPanelDimension downPanelDimension) {
		GUIManager.downPanelDimension = downPanelDimension;
	}

	public static LeftPanelDimension getLeftPanelDimension() {
		if (leftPanelDimension == null) {
			leftPanelDimension = new LeftPanelDimension();
		}
		return leftPanelDimension;
	}

	public static void setLeftPanelDimension(LeftPanelDimension leftPanelDimension) {
		GUIManager.leftPanelDimension = leftPanelDimension;
	}

	public static ShowPanel getShowPanel() {
		if (showPanel == null) {
			showPanel = new ShowPanel();
		}
		return showPanel;
	}

	public static void setShowPanel(ShowPanel showPanel) {
		GUIManager.showPanel = showPanel;
	}

	/**
	 * @return the maqsudPanel
	 */
	public static CalculationPanel getCalculationPanel() {
		if (calculationPanel == null) {
			calculationPanel = new CalculationPanel();
		}
		return calculationPanel;
	}

	/**
	 * @param maqsudPanel
	 *            the maqsudPanel to set
	 */
	public static void setCalculationPanel(CalculationPanel calculationPanel) {
		GUIManager.calculationPanel = calculationPanel;
	}

}