/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @author Hooman
 */
public class MassSpring extends JPanel {

    static Point[][] points = new Point[100][100];
    static Spring[][] springsVer = new Spring[101][101];
    static Spring[][] springsHor = new Spring[101][101];
    static Point[][] points2 = new Point[100][100];
    static Spring[][] springsVer2 = new Spring[101][101];
    static Spring[][] springsHor2 = new Spring[101][101];


    public static void main(String[] args) {
        for (int i = 0; i < 101; i++) {
            for (int j = 0; j < 101; j++) {
                springsVer[i][j] = new Spring(.001);
                springsHor[i][j] = new Spring(.001);
            }
        }
        for (int i = 0; i < 101; i++) {
            for (int j = 0; j < 101; j++) {
                springsVer2[i][j] = new Spring(.001);
                springsHor2[i][j] = new Spring(.001);
            }
        }
// test case 1 comment case 2 to run this        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                points[i][j] = new Point(1, j + 1, i + 1, 3, 0, 0, 0, springsVer[i][j], springsVer[i + 1][j], springsHor[i][j], springsHor[i][j + 1]);
                points2[i][j] = new Point(1, 899 + j, i + 1, -3, 0, 0, 0, springsVer2[i][j], springsVer2[i + 1][j], springsHor2[i][j], springsHor2[i][j + 1]);
            }
        }

//  test case 2 uncomment and comment case 1
//        for (int i = 0; i < 100; i++) {
//            for (int j = 0; j < 50; j++) {
//                points[i][j] = new Point(1, j + 1, i + 1, 1, 0, 0, 0, springsVer[i][j], springsVer[i + 1][j], springsHor[i][j], springsHor[i][j + 1]);
//                points2[i][j] = new Point(1, j + 899, i + 1, -2, 0, 0, 0, springsVer2[i][j], springsVer2[i + 1][j], springsHor2[i][j], springsHor2[i][j + 1]);
//            }
//            for (int j = 50; j < 100; j++) {
//                points[i][j] = new Point(1, j + 1, i + 1, 2, 0, 0, 0, springsVer[i][j], springsVer[i + 1][j], springsHor[i][j], springsHor[i][j + 1]);
//                points2[i][j] = new Point(1, j + 899, i + 1, -1, 0, 0, 0, springsVer2[i][j], springsVer2[i + 1][j], springsHor2[i][j], springsHor2[i][j + 1]);
//            }
//        }

//

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {

                if (i >= 1) {
                    springsVer[i][j].left = points[i - 1][j];
                    springsVer[i][j].right = points[i][j];
                    springsVer2[i][j].left = points2[i - 1][j];
                    springsVer2[i][j].right = points2[i][j];
                } else {
                    springsVer[i][j].left = points[i][j];
                    springsVer[i][j].right = points[i][j];
                    springsVer2[i][j].left = points2[i][j];
                    springsVer2[i][j].right = points2[i][j];
                }

                if (j >= 1) {
                    springsHor[i][j].left = points[i][j - 1];
                    springsHor[i][j].right = points[i][j];
                    springsHor2[i][j].left = points2[i][j - 1];
                    springsHor2[i][j].right = points2[i][j];
                } else {
                    springsHor[i][j].left = points[i][j];
                    springsHor[i][j].right = points[i][j];
                    springsHor2[i][j].left = points2[i][j];
                    springsHor2[i][j].right = points2[i][j];
                }
                if (i == 99) {
                    springsVer[i + 1][j].left = points[i][j];
                    springsVer[i + 1][j].right = points[i][j];
                    springsVer2[i + 1][j].left = points2[i][j];
                    springsVer2[i + 1][j].right = points2[i][j];
                }
                if (j == 99) {
                    springsHor[i][j + 1].left = points[i][j];
                    springsHor[i][j + 1].right = points[i][j];
                    springsHor2[i][j + 1].left = points2[i][j];
                    springsHor2[i][j + 1].right = points2[i][j];
                }
            }
        }
        JFrame.setDefaultLookAndFeelDecorated(true);
        JFrame frame = new JFrame("Simulation");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setBackground(Color.white);
        frame.setSize(1000, 600);
        frame.setLocation((Toolkit.getDefaultToolkit().getScreenSize().width - 1000) / 2, (Toolkit.getDefaultToolkit().getScreenSize().height - 600) / 2);

        MassSpring panel = new MassSpring();

        frame.add(panel);

        frame.setVisible(true);
    }

    public void paint(Graphics g) {

//        g.setColor(Color.red);
//        for (int count = 1; count < 2; count++) {

        g.clearRect(0, 0, 999, 599);
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                points[i][j].move(1);
                points2[i][j].move(1);
                //g.drawLine(10,10,10,100);
//                    g.setColor(Color.blue);
                if (i % 10 == 0 && j % 10 == 0) {
                    if (springsHor[i][j].torn == false) {
                        g.setColor(Color.blue);
                        g.drawLine((int) springsHor[i][j].left.x, (int) (400 - springsHor[i][j].left.y), (int) springsHor[i][j].right.x, (int) (400 - springsHor[i][j].right.y));
                    }
                    if (springsHor2[i][j].torn == false) {
                        g.setColor(Color.red);
                        g.drawLine((int) springsHor2[i][j].left.x, (int) (400 - springsHor2[i][j].left.y), (int) springsHor2[i][j].right.x, (int) (400 - springsHor2[i][j].right.y));
                    }
                    if (springsVer[i][j].torn == false) {
                        g.setColor(Color.blue);
                        g.drawLine((int) springsVer[i][j].left.x, (int) (400 - springsVer[i][j].left.y), (int) springsVer[i][j].right.x, (int) (400 - springsVer[i][j].right.y));
                    }
                    if (springsVer2[i][j].torn == false) {
                        g.setColor(Color.red);
                        g.drawLine((int) springsVer2[i][j].left.x, (int) (400 - springsVer2[i][j].left.y), (int) springsVer2[i][j].right.x, (int) (400 - springsVer2[i][j].right.y));
                    }
                }
                g.setColor(Color.red);
                g.drawLine((int) points[i][j].x, (int) (400 - points[i][j].y), (int) points[i][j].x, (int) (400 - points[i][j].y));
                g.setColor(Color.blue);
                g.drawLine((int) points2[i][j].x, (int) (400 - points2[i][j].y), (int) points2[i][j].x, (int) (400 - points2[i][j].y));


            }

            try {
                TimeUnit.MILLISECONDS.sleep(1);
            } catch (InterruptedException ex) {
                Logger.getLogger(MassSpring.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j++) {
                Point one = points[i][j];
                for (int k = 0; k < 100; k++)
                    for (int z = 0; z < 100; z++) {
                        Point two = points2[k][z];
                        if (one.x == two.x && one.y == two.y)
                            Point.collision(one, two);
                    }
            }


        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j++) {
                points[i][j].update(1);
                points2[i][j].update(1);
            }
        repaint();
//        }
    }

}
