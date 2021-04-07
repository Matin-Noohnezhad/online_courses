package org.bihe;

import org.bihe.production.Production;

import java.util.ArrayList;

public class ContextFreeGrammar {

    private ArrayList<Variable> variables;
    private ArrayList<Terminal> terminals;
    private ArrayList<Production> productions;
    private Variable startVariable;

    public ContextFreeGrammar(ArrayList<Variable> variables, ArrayList<Terminal> terminals, ArrayList<Production> productions, Variable startVariable) {
        this.variables = variables;
        this.terminals = terminals;
        this.productions = productions;
        this.startVariable = startVariable;
    }

    public ArrayList<Variable> getVariables() {
        return variables;
    }

    public void setVariables(ArrayList<Variable> variables) {
        this.variables = variables;
    }

    public ArrayList<Terminal> getTerminals() {
        return terminals;
    }

    public void setTerminals(ArrayList<Terminal> terminals) {
        this.terminals = terminals;
    }

    public ArrayList<Production> getProductions() {
        return productions;
    }

    public void setProductions(ArrayList<Production> productions) {
        this.productions = productions;
    }

    public Variable getStartVariable() {
        return startVariable;
    }

    public void setStartVariable(Variable startVariable) {
        this.startVariable = startVariable;
    }
}
