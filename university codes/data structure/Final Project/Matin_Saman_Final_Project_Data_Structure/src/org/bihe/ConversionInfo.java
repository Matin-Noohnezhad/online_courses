package org.bihe;

public class ConversionInfo {

    private int currency1;
    private int currency2;
    private double conversionRate;
    private double rateLog;
    private double sumPathLog;
    private String shortestPath;


    public ConversionInfo(int currency1, int currency2, double conversionRate) {
        this.currency1 = currency1;
        this.currency2 = currency2;
        this.conversionRate = conversionRate;
        rateLog = Math.log(conversionRate);

        sumPathLog = rateLog;
        shortestPath = "-->";
    }

    public double getRateLog() {
        return rateLog;
    }

    public void setRateLog(double rateLog) {
        this.rateLog = rateLog;
    }

    public int getCurrency1() {
        return currency1;
    }

    public void setCurrency1(int currency1) {
        this.currency1 = currency1;
    }

    public int getCurrency2() {
        return currency2;
    }

    public void setCurrency2(int currency2) {
        this.currency2 = currency2;
    }

    public double getConversionRate() {
        return conversionRate;
    }

    public void setConversionRate(double conversionRate) {
        this.conversionRate = conversionRate;
    }

    public double getSumPathLog() {
        return sumPathLog;
    }

    public void setSumPathLog(double sumPathLog) {
        this.sumPathLog = sumPathLog;
    }

    public String getShortestPath() {
        return shortestPath;
    }

    public void setShortestPath(String shortestPath) {
        this.shortestPath = shortestPath;
    }
}
