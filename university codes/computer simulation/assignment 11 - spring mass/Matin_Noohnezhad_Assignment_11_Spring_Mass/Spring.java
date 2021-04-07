/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * @author Hooman
 */
public class Spring {
    double k;
    Point left;
    Point right;
    int tornLength = 500;
    boolean torn = false;

    public Spring(double k, Point left, Point right) {
        this.k = k;
        this.left = left;
        this.right = right;
    }

    public Spring(double k) {
        this.k = k;
    }

    public double getForceX() {
        double length = Math.sqrt(Math.pow(left.x - right.x, 2) + Math.pow(left.y - right.y, 2));
        if (torn || (length > tornLength)) {
            torn = true;
            return 0;
        }
        if (Math.abs(right.x - left.x) > 1) {
            return (right.x - left.x) * k;
        }
        return 0;
    }

    public double getForceY() {
        double length = Math.sqrt(Math.pow(left.x - right.x, 2) + Math.pow(left.y - right.y, 2));
        if (torn || (length > tornLength)) {
            torn = true;
            return 0;
        }
        if (Math.abs(right.y - left.y) > 1) {
            return (right.y - left.y) * k;
        }
        return 0;
    }

}
