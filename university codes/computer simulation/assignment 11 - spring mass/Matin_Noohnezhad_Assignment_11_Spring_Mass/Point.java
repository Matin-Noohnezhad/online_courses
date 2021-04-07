/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * @author Hooman
 */
public class Point {

    double m;
    double x;
    double y;
    double aX;
    double aY;
    double vX;
    double vY;
    double fX;
    double fY;
    Spring up;
    Spring down;
    Spring left;
    Spring right;

    public Point(double m, double x, double y, double vX, double vY, double fX, double fY, Spring up, Spring down, Spring left, Spring right) {
        this.m = m;
        this.x = x;
        this.y = y;
        this.aX = fX / m;
        this.aY = fY / m;
        this.vX = vX;
        this.vY = vY;
        this.fX = fX;
        this.fY = fY;
        this.up = up;
        this.down = down;
        this.left = left;
        this.right = right;
    }

    public void move(double time) {
        x = x + .5 * aX * time * time + vX * time;
        y = y + .5 * aY * time * time + vY * time;
    }

    public void update(double time) {
        fX = right.getForceX() - left.getForceX();
        fY = down.getForceY() - up.getForceY();
        aX = fX / m;
        aY = fY / m;
        vX = vX + aX * time;
        vY = vY + aY * time;
    }

    public static void collision(Point one, Point two) {
        double m1 = one.m;
        double m2 = two.m;
        double v1X = one.vX;
        double v2X = two.vX;
        double v1Y = one.vY;
        double v2Y = two.vY;

        one.vX = (m1 - m2) / (m1 + m2) * v1X + (2 * m2) / (m1 + m2) * v2X;
        two.vX = (m2 - m1) / (m1 + m2) * v2X + (2 * m1) / (m1 + m2) * v1X;
        one.vY = (m1 - m2) / (m1 + m2) * v1Y + (2 * m2) / (m1 + m2) * v2Y;
        two.vY = (m2 - m1) / (m1 + m2) * v2Y + (2 * m1) / (m1 + m2) * v1Y;

    }
}