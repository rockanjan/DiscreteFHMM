package model.param;

import java.util.Random;

import config.Config;

public class LogLinearWeightsClass {
	public double[][] weights; //weights for the log-linear model
	
	public int M; //layers
	public int K; //states
	public int C; //classes
	
	public LogLinearWeightsClass(int m, int k, int c) {
		this.M = m;
		this.K = k;
		this.C = c;
		weights = new double[c][m*k];
	}
	
	public double[] getStateVector(int m, int c) {
		double[] vector = new double[Config.numStates];
		for(int i=0; i<K; i++) {
			vector[i] = weights[c][m*K + i];
		}
		return vector;
	}
	
	public static int getIndex(int m, int k) {
		return m * Config.numStates + k;
	}
	
	public double get(int m, int k, int v) {
		return weights[v][m * K + k];	
	}
	
	public void set(int m, int k, int v,  double value) {
		weights[v][m * Config.numStates + k] = value;	
	}
	
	public void initializeZeros() {
		weights = new double[C][M * K]; 
	}
	
	public void initializeUniform(double value) {
		weights = new double[C][M * K];
		for(int y=0; y<C; y++) {
			for(int u=0; u<M*K; u++) {
				weights[y][u] = value;
			}
		}
	}
	
	public void initializeRandom(Random r) {
		weights = new double[C][M*K];
		for(int y=0; y<C; y++) {
			for(int u=0; u<M*K; u++) {
				weights[y][u] = r.nextDouble();
			}
		}
	}	
	
	/*
	 * Clone with weights exp
	 */
	public LogLinearWeightsClass getCloneExp() {
		LogLinearWeightsClass clone = new LogLinearWeightsClass(M, K, C);
		clone.initializeZeros();
		for(int i=0; i<C; i++) {
			for(int j=0; j<M*K; j++) {
				clone.weights[i][j] = Math.exp(this.weights[i][j]);
			}
		}
		return clone;
	}
	
	public void cloneFrom(LogLinearWeightsClass source) {
		for(int i=0; i<C; i++) {
			for(int j=0; j<M*K; j++) {
				this.weights[i][j] = source.weights[i][j];
			}
		}
	}
	
	public boolean equalsExact(LogLinearWeightsClass other) {
		boolean result = true;
		if(other.weights.length != weights.length || other.weights[0].length != weights[0].length) {
			result = false;
			return result;
		}
		for(int i=0; i<weights.length; i++) {
			for(int j=0; j<weights[0].length; j++) {
				if(weights[i][j] != other.weights[i][j]) {
					result = false;
					break;
					
				}
			}
		}
		return result;
	}
	
	public boolean equalsApprox(LogLinearWeightsClass other) {
		boolean result = true;
		if(other.weights.length != weights.length || other.weights[0].length != weights[0].length) {
			result = false;
			return result;
		}
		for(int i=0; i<weights.length; i++) {
			for(int j=0; j<weights[0].length; j++) {
				if(Math.abs(weights[i][j] - other.weights[i][j]) > 1e-5) {
					result = false;
					break;
					
				}
			}
		}
		return result;
	}
}
