package model.param;

import java.util.Arrays;
import java.util.Random;

public class LogLinearWeightsClass {
	public double[] weights; //weights for the log-linear model
	public static int[] states;
	public int C; //classes
	public static int length;
	public static int sumKOverAllLayers;
	public LogLinearWeightsClass(int c, int[] states) {
		this.states = states;
		this.C = c;
		sumKOverAllLayers = 0;
		for(int m=0; m<states.length; m++) {
			sumKOverAllLayers += states[m];
		}
		length = c * sumKOverAllLayers; 
		//System.out.println("classlength = " + length);
		weights = new double[length];
	}
	
	public double[] getStateVector(int m, int c) {
		int startIndex = getIndex(m, 0, c);
		return Arrays.copyOfRange(weights, startIndex, startIndex + states[m]);
	}
	
	/*
	 * for C
	 * 	  for m
	 * 		for k
	 */
	public static int getIndex(int m, int k, int c) {
		int sumKUptom = 0;
		for(int n=0; n<m; n++) {
			sumKUptom += states[n];
		}
		return c * sumKOverAllLayers + sumKUptom + k;
	}
	
	public double get(int m, int k, int c) {
		//System.out.format("m=%d,k=%d,c=%d\n",m,k,c);
		return weights[getIndex(m, k, c)];	
	}
	
	public void set(int m, int k, int c,  double value) {
		weights[getIndex(m, k, c)] = value;	
	}
	
	public void initializeZeros() {
		weights = new double[length]; 
	}
	
	public void initializeUniform(double value) {
		weights = new double[length];
		for(int y=0; y<length; y++) {
			weights[y] = value;
		}
	}
	
	public void initializeRandom(Random r) {
		weights = new double[length];
		for(int y=0; y<length; y++) {
			weights[y] = r.nextDouble();
		}
	}	
	
	/*
	 * Clone with weights exp
	 */
	public LogLinearWeightsClass getCloneExp() {
		LogLinearWeightsClass clone = new LogLinearWeightsClass(C, states);
		clone.initializeZeros();
		for(int i=0; i<length; i++) {
			clone.weights[i] = Math.exp(this.weights[i]);			
		}
		return clone;
	}
	
	public void cloneFrom(LogLinearWeightsClass source) {
		for(int i=0; i<length; i++) {
			this.weights[i] = source.weights[i];			
		}
	}
	
	public boolean equalsExact(LogLinearWeightsClass other) {
		boolean result = true;
		if(other.weights.length != weights.length) {
			result = false;
			return result;
		}
		for(int i=0; i<weights.length; i++) {
			if(weights[i] != other.weights[i]) {
				result = false;
				break;
			}
		}
		return result;
	}
	
	public boolean equalsApprox(LogLinearWeightsClass other) {
		boolean result = true;
		if(other.weights.length != weights.length) {
			result = false;
			return result;
		}
		for(int i=0; i<weights.length; i++) {			
			if(Math.abs(weights[i] - other.weights[i]) > 1e-5) {
				result = false;
				break;
			}
		}
		return result;
	}
}
