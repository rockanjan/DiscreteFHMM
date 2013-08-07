package model.param;

import java.util.Random;

import program.Main;

public class LogLinearWeights {
	public double[][] weights; //weights for the log-linear model
	
	public int vocabSize;
	public static int conditionalSize;
	
	/*
	 * @param vocabSize = number of distinct observation tokens
	 * @param conditionSize = dimension of current hidden layer, |X|  
	 * and sum of dimensions of previous decodes states |Z_vector|
	 */
	public LogLinearWeights(int vocabSize, int xzSize) {
		this.vocabSize = vocabSize;
		//this.conditionalSize = xzSize + 1;		 //1 for offset
		this.conditionalSize = xzSize;
	}
	
	public double[] getStateVector(int m, int v) {
		double[] vector = new double[Main.numStates];
		for(int i=0; i<Main.numStates; i++) {
			vector[i] = weights[v][m*Main.numStates + i];
		}
		return vector;
	}
	
	public double get(int m, int k, int v) {
		//return weights[v][m*k + k];
		return weights[v][m*Main.numStates + k];	
	}
	public void initializeZeros() {
		weights = new double[vocabSize][conditionalSize]; 
	}
	
	public void initializeUniform(double value) {
		weights = new double[vocabSize][conditionalSize];
		for(int y=0; y<vocabSize; y++) {
			for(int u=0; u<conditionalSize; u++) {
				weights[y][u] = value;
			}
		}
	}
	
	public void initializeRandom(Random r) {
		weights = new double[vocabSize][conditionalSize];
		for(int y=0; y<vocabSize; y++) {
			for(int u=0; u<conditionalSize; u++) {
				//weights[y][u] = r.nextGaussian();
				weights[y][u] = r.nextDouble() * .1;
			}
		}
	}
	
	/*
	public LogLinearWeights getClone() {
		LogLinearWeights clone = new LogLinearWeights(vocabSize, conditionalSize);
		clone.initializeZeros();
		for(int i=0; i<vocabSize; i++) {
			for(int j=0; j<conditionalSize; j++) {
				clone.weights[i][j] = this.weights[i][j];
			}
		}
		return clone;
	}
	*/
	
	/*
	 * Clone with weights exp
	 */
	public LogLinearWeights getCloneExp() {
		LogLinearWeights clone = new LogLinearWeights(vocabSize, conditionalSize);
		clone.initializeZeros();
		for(int i=0; i<vocabSize; i++) {
			for(int j=0; j<conditionalSize; j++) {
				clone.weights[i][j] = Math.exp(this.weights[i][j]);
			}
		}
		return clone;
	}
	
	public void cloneFrom(LogLinearWeights source) {
		for(int i=0; i<vocabSize; i++) {
			for(int j=0; j<conditionalSize; j++) {
				this.weights[i][j] = source.weights[i][j];
			}
		}
	}
	
	public boolean equalsExact(LogLinearWeights other) {
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
	
	public boolean equalsApprox(LogLinearWeights other) {
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
