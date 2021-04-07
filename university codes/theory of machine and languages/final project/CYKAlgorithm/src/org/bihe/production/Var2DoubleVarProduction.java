package org.bihe.production;

import org.bihe.Variable;

public class Var2DoubleVarProduction extends Production{

    private Variable rightVariable1;
    private Variable rightVariable2;

    public Var2DoubleVarProduction(Variable leftVar, Variable rightVar1,Variable rightVar2, boolean isVar2Var){
        super(leftVar,isVar2Var);
        this.rightVariable1 = rightVar1;
        this.rightVariable2 = rightVar2;
    }

    public Variable getRightVariable1() {
        return rightVariable1;
    }

    public void setRightVariable1(Variable rightVariable1) {
        this.rightVariable1 = rightVariable1;
    }

    public Variable getRightVariable2() {
        return rightVariable2;
    }

    public void setRightVariable2(Variable rightVariable2) {
        this.rightVariable2 = rightVariable2;
    }
}
