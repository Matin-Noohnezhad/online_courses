package org.bihe.main;

public class GUIManager {

	private static MainFrame mainFrame;
	private static LoginPanel loginPanel;
	private static EnrollmentPanel enrollmentPanel;
	private static AdminPanel adminPanel;

	public static MainFrame getMainFrame() {
		if (mainFrame == null) {
			mainFrame = new MainFrame();
		}
		return mainFrame;
	}

	public static LoginPanel getLoginPanel() {
		if (loginPanel == null) {
			loginPanel = new LoginPanel();
		}
		return loginPanel;
	}

	public static EnrollmentPanel getEnrollmentPanel() {
		if (enrollmentPanel == null) {
			enrollmentPanel = new EnrollmentPanel();
		}
		return enrollmentPanel;
	}

	public static AdminPanel getAdminPanel() {
		if (adminPanel == null) {
			adminPanel = new AdminPanel();
		}
		return adminPanel;
	}

	public static void addLoginPanelToCenter() {
		mainFrame = getMainFrame();
		loginPanel = getLoginPanel();

		mainFrame.addToCenter(loginPanel);
	}

	public static void addEnrollmentPanelToCenter() {
		mainFrame = getMainFrame();
		enrollmentPanel = getEnrollmentPanel();

		mainFrame.addToCenter(enrollmentPanel);
	}

	public static void addAdminPanelToCenter() {
		mainFrame = getMainFrame();
		adminPanel = getAdminPanel();
		
		mainFrame.addToCenter(adminPanel);
	}

}
