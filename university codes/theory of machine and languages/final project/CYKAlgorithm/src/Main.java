import org.bihe.ContextFreeGrammar;
import org.bihe.Terminal;
import org.bihe.Variable;
import org.bihe.algorithm.Algorithm;
import org.bihe.production.Production;
import org.bihe.production.Var2DoubleVarProduction;
import org.bihe.production.Var2TerProduction;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        buildTest();
    }

    private static void buildTest() {
        Variable S = new Variable("S");
        Variable B = new Variable("B");
        Variable C = new Variable("C");
        Variable D = new Variable("D");
        Variable E = new Variable("E");
        Variable X = new Variable("X");
        ArrayList<Variable> variables = new ArrayList<>();
        variables.add(S);
        variables.add(B);
        variables.add(C);
        variables.add(D);
        variables.add(E);
        variables.add(X);
        //
        Terminal x = new Terminal("x");
        Terminal y = new Terminal("y");
        Terminal z = new Terminal("z");
        Terminal leftPar = new Terminal("(");
        Terminal rightPar = new Terminal(")");
        Terminal mult = new Terminal("*");
        Terminal sum = new Terminal("+");
        Terminal minus = new Terminal("-");
        Terminal div = new Terminal("/");
        ArrayList<Terminal> terminals = new ArrayList<>();
        terminals.add(x);
        terminals.add(y);
        terminals.add(z);
        terminals.add(leftPar);
        terminals.add(rightPar);
        terminals.add(mult);
        terminals.add(sum);
        terminals.add(minus);
        terminals.add(div);
        //
        Var2DoubleVarProduction pro1 = new Var2DoubleVarProduction(S, B, D, true);
        Var2DoubleVarProduction pro2 = new Var2DoubleVarProduction(D, S, C, true);
        Var2DoubleVarProduction pro3 = new Var2DoubleVarProduction(S, S, E, true);
        Var2DoubleVarProduction pro4 = new Var2DoubleVarProduction(E, X, S, true);
        Var2TerProduction pro5 = new Var2TerProduction(S, x, false);
        Var2TerProduction pro6 = new Var2TerProduction(S, y, false);
        Var2TerProduction pro7 = new Var2TerProduction(S, z, false);
        Var2TerProduction pro8 = new Var2TerProduction(X, mult, false);
        Var2TerProduction pro9 = new Var2TerProduction(X, sum, false);
        Var2TerProduction pro10 = new Var2TerProduction(X, minus, false);
        Var2TerProduction pro11 = new Var2TerProduction(X, div, false);
        Var2TerProduction pro12 = new Var2TerProduction(B, leftPar, false);
        Var2TerProduction pro13 = new Var2TerProduction(C, rightPar, false);
        ArrayList<Production> productions = new ArrayList<>();
        productions.add(pro1);
        productions.add(pro2);
        productions.add(pro3);
        productions.add(pro4);
        productions.add(pro5);
        productions.add(pro6);
        productions.add(pro7);
        productions.add(pro8);
        productions.add(pro9);
        productions.add(pro10);
        productions.add(pro11);
        productions.add(pro12);
        productions.add(pro13);
        //
        ContextFreeGrammar cfg = new ContextFreeGrammar(variables, terminals, productions, S);
        //this machine cant handle space between words. be carefull!!!
        String w = "(x-(y+z)*(x-y))+z";
        printOutput(w, Algorithm.CYKAlgorithm(cfg, w));
        String w2 = "x+x*y(z*)";
        printOutput(w2, Algorithm.CYKAlgorithm(cfg, w2));
        String w3 = "(z+x*y";
        printOutput(w3, Algorithm.CYKAlgorithm(cfg, w3));


    }

    private static void printOutput(String w, boolean accepted) {
        if (accepted) {
            System.out.println(w + " is accepted by this machine.");
        } else {
            System.out.println(w + " is rejected by this machine.");
        }
    }

}
