/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator3;

import java.io.IOException;

/**
 *
 * @author Hooman
 */
public class Simulator2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        int [] p={1,1,1,1,1,1,1,1};
        for(int i=2;i<=22;i+=2)
        {
           for(int j=2;j<=22;j+=2)
           {
               for(int k=2;k<=22;k+=2)
               {
                   for(int l=2;l<=22;l+=2)
                   {
                        Executer ex=new Executer(1000000,p);
                        ex.simualte();
                        System.out.println(p[1]);
                        p[7]=l;
                   }
                   p[5]=k;
               }
               p[3]=j;
           }
           p[1]=i;
        }
    }
    
}
