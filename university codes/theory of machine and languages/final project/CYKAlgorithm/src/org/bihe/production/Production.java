package org.bihe.production;

import org.bihe.Variable;

public class Production {

    private Variable leftVariable;
    private boolean isVar2Var;

    public Production(Variable leftVar,boolean isVar2Var){
        this.leftVariable = leftVar;
        this.isVar2Var = isVar2Var;
    }

    public Variable getLeftVariable() {
        return leftVariable;
    }

    public void setLeftVariable(Variable leftVariable) {
        this.leftVariable = leftVariable;
    }

    public boolean isVar2Var() {
        return isVar2Var;
    }

    public void setVar2Var(boolean var2Var) {
        isVar2Var = var2Var;
    }
}
