package calculations;

import java.util.ArrayList;

import bean.ElectricCharge;

public class Calculate {
	private static double distance(double x1, double y1, double x2, double y2) {
		return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
	}

	public static double electricPotential(ArrayList<ElectricCharge> electricCharges, int x, int y) {
		double resualt = 0;
		double temp;
		for (int i = 0; i < electricCharges.size(); i++) {
			temp = electricCharges.get(i).getCharge() / (4 * 3.1415 * 8.854187 * Math.pow(10, -11)
					* distance(electricCharges.get(i).getX(), electricCharges.get(i).getY(), x, y));
			resualt = resualt + temp;
		}
		return resualt;
	}

	public static double angele(int x1, int y1, int x2, int y2) {
		double ab = distance(x1, y1, x2, y2);
		double ac = distance(x1, y1, x2, y1);

		double angele = Math.acos(ac / ab);
		return angele;

	}

	public static double electricFiled(ElectricCharge electricCharge, int x, int y) {
		double resualt;

		resualt = electricCharge.getCharge() / (4 * 3.1415 * 8.854187 * Math.pow(10, -11)
				* Math.pow(distance(electricCharge.getX(), electricCharge.getY(), x, y), 2));

		return resualt;
	}

	public static double electricFiledX(ElectricCharge electricCharge, int x, int y) {
		if (electricCharge.getX() < x) {
			return electricFiled(electricCharge, x, y)
					* Math.cos(angele(electricCharge.getX(), electricCharge.getY(), x, y));

		} else
			return -electricFiled(electricCharge, x, y)
					* Math.cos(angele(electricCharge.getX(), electricCharge.getY(), x, y));
	}

	public static double electricFiledY(ElectricCharge electricCharge, int x, int y) {
		if (electricCharge.getY() > y) {
			return -electricFiled(electricCharge, x, y)
					* Math.sin(angele(electricCharge.getX(), electricCharge.getY(), x, y));
		} else
			return electricFiled(electricCharge, x, y)
					* Math.sin(angele(electricCharge.getX(), electricCharge.getY(), x, y));
	}

	public static double electricFieldsX(ArrayList<ElectricCharge> electricCharges, int x, int y) {
		double resualt = 0;
		double temp;
		for (int i = 0; i < electricCharges.size(); i++) {
			temp = electricFiledX(electricCharges.get(i), x, y);
			resualt = resualt + temp;
		}
		return resualt;
	}

	public static double electricFieldsY(ArrayList<ElectricCharge> electricCharges, int x, int y) {
		double resualt = 0;
		double temp;
		for (int i = 0; i < electricCharges.size(); i++) {
			temp = electricFiledY(electricCharges.get(i), x, y);
			resualt = resualt + temp;
		}
		return resualt;
	}
}
