package org.bihe.algorithm;

import org.bihe.ContextFreeGrammar;
import org.bihe.Variable;
import org.bihe.production.Production;
import org.bihe.production.Var2DoubleVarProduction;
import org.bihe.production.Var2TerProduction;

import java.util.ArrayList;

public class Algorithm {

    public static boolean CYKAlgorithm(ContextFreeGrammar cfg, String w) {
        int n = w.length();
        //
        ArrayList<Variable>[][] X = new ArrayList[n][n];
        for(int i =0;i<n;i++){
            for(int j =0;j<n;j++){
                X[i][j]= new ArrayList<Variable>();
            }
        }

        //first part
        for (int i = 0; i < n; i++) {
            for (Variable variable : cfg.getVariables()) {
//                boolean exist = false;
                for (Production production : cfg.getProductions()) {
                    if (!production.isVar2Var()) {
                        if (((Var2TerProduction) production).getLeftVariable().equals(variable)) {
                            if (((Var2TerProduction) production).getTerminal().getTerminalName().equals(w.substring(i, i + 1))) {
                                X[i][i].add(variable);
                            }
                        }
                    }
                }
            }
        }
        //first part(end)
        //second part
        for (int s = 1; s < n; s++) {
            for (int i = 0; i < n - s; i++) {
                for (int k = i; k < i + s; k++) {
                    for (Variable B : X[i][k]) {
                        for (Variable C : X[k + 1][i + s]) {
                            for (Variable A : cfg.getVariables()) {
                                for (Production production : cfg.getProductions()) {
                                    if (production.isVar2Var()) {
                                        Var2DoubleVarProduction p = (Var2DoubleVarProduction) production;
                                        if (p.getLeftVariable().equals(A) && p.getRightVariable1().equals(B) && p.getRightVariable2().equals(C)) {
                                            X[i][i + s].add(A);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        //second part(end)
        boolean answer = false;
        for (Variable s : X[0][n - 1]) {
            if (s.equals(cfg.getStartVariable())) {
                answer = true;
            }
        }
        return answer;
    }

}
