package org.bihe;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;

/**
 * 
 * @author Farzam This class stores the information of each action (method call)
 *         in FileUtil class
 */
public class Log implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4027456649914718517L;
	
	/**
	 * When method call started.
	 */
	private Date startActionDate;

	/**
	 * How long did the method call take. (in milliseconds)
	 */
	private long timeOfAction;

	/**
	 * What method has been called.
	 */
	private MethodNames methodCall;

	/**
	 * Information that is produced by method call.
	 */
	private ArrayList<String> moreInfo = new ArrayList<>();

	// Constructors
	// ***************************************
	public Log(Date startActionDate, long timeOfAction, MethodNames methodCall, ArrayList<String> moreInfo) {
		super();
		this.startActionDate = startActionDate;
		this.timeOfAction = timeOfAction;
		this.methodCall = methodCall;
		this.moreInfo = moreInfo;
	}

	public Log(Date startActionDate, MethodNames methodCall, ArrayList<String> moreInfo) {
		super();
		this.startActionDate = startActionDate;
		this.methodCall = methodCall;
		this.moreInfo = moreInfo;
	}

	public Log() {

	}

	/**
	 * @return the startActionDate
	 */
	public Date getStartActionDate() {
		return startActionDate;
	}

	/**
	 * @param startActionDate
	 *            the startActionDate to set
	 */
	public void setStartActionDate(Date startActionDate) {
		this.startActionDate = startActionDate;
	}

	/**
	 * @return the timeOfAction
	 */
	public long getTimeOfAction() {
		return timeOfAction;
	}

	/**
	 * @param timeOfAction
	 *            the timeOfAction to set
	 */
	public void setTimeOfAction(long timeOfAction) {
		this.timeOfAction = timeOfAction;
	}

	/**
	 * @return the methodCall
	 */
	public MethodNames getMethodCall() {
		return methodCall;
	}

	/**
	 * @param methodCall
	 *            the methodCall to set
	 */
	public void setMethodCall(MethodNames methodCall) {
		this.methodCall = methodCall;
	}

	/**
	 * @return the moreInfo
	 */
	public ArrayList<String> getMoreInfo() {
		return moreInfo;
	}

	/**
	 * @param moreInfo
	 *            the moreInfo to set
	 */
	public void setMoreInfo(ArrayList<String> moreInfo) {
		this.moreInfo = moreInfo;
	}

	// Methods
	// ***************************************
	/**
	 * 
	 * @param info
	 *            : the info to add to previous informations
	 */
	public void addMoreInfo(String info) {
		this.moreInfo.add(info);
	}

	@Override
	public String toString() {
		// TODO
		// Implement this method (according to what you need to save)
		return super.toString();
	}
}
