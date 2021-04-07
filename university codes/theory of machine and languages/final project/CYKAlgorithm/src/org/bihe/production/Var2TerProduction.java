package org.bihe.production;

import org.bihe.Terminal;
import org.bihe.Variable;

public class Var2TerProduction extends Production{

    private Terminal terminal;

    public Var2TerProduction(Variable leftVar,Terminal terminal, boolean isVar2Var){
        super(leftVar,isVar2Var);
        this.terminal = terminal;
    }

    public Terminal getTerminal() {
        return terminal;
    }

    public void setTerminal(Terminal terminal) {
        this.terminal = terminal;
    }
}
