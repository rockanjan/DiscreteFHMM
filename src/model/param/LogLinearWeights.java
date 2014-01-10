package model.param;

import java.util.Arrays;
import java.util.Random;

public class LogLinearWeights {
	public double[] weights; //weights for the log-linear model
	public int vocabSize;
	public static int[] states;
	public static int length;
	public static int sumKOverAllLayers;
	public LogLinearWeights(int vocabSize, int[] states) {
		this.vocabSize = vocabSize;
		this.states = states;
		sumKOverAllLayers = 0;
		for(int m=0; m<states.length; m++) {
			sumKOverAllLayers += states[m];
		}
		length = vocabSize * sumKOverAllLayers; 
		System.out.println("wordlength = " + length);
	}
	
	public double[] getStateVector(int m, int v) {
		int startIndex = getIndex(m, 0, v);
		return Arrays.copyOfRange(weights, startIndex, startIndex + states[m]);
	}
	
	/*
	 * for V
	 * 	  for m
	 * 		for k
	 * then, index = v * (sum_m' K_m') + sum_{m' < m} K_m' + k
	 */
	public static int getIndex(int m, int k, int v) {
		int sumKUptom = 0;
		for(int n=0; n<m; n++) {
			sumKUptom += states[n];
		}
		return v * sumKOverAllLayers + sumKUptom + k;
		
	}
	
	public double get(int m, int k, int v) {
		return weights[getIndex(m, k, v)];	
	}
	
	public void set(int m, int k, int v,  double value) {
		weights[getIndex(m, k, v)] = value;	
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
	public void initializeFromDifferentLayerModel(LogLinearWeights source) {
		int sourceNrLayers = source.weights[0].length / Config.numStates;
		int thisNrLayers = this.weights[0].length / Config.numStates;
		if(sourceNrLayers < thisNrLayers) {
			for(int m=0; m<sourceNrLayers; m++) {
				for(int k=0; k<Config.numStates; k++) {
					for(int v=0; v<weights.length; v++) {
						this.set(m,k,v, source.get(m,k,v));
					}
				}
			}
			//other layers will remain the same as in initialization
		} else {
			for(int m=0; m<thisNrLayers; m++) {
				for(int k=0; k<Config.numStates; k++) {
					for(int v=0; v<weights.length; v++) {
						this.set(m,k,v, source.get(m,k,v));
					}
				}
			}
		}
	}
	*/
	
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
		LogLinearWeights clone = new LogLinearWeights(vocabSize, states);
		clone.initializeZeros();
		for(int i=0; i<length; i++) {
			clone.weights[i] = Math.exp(this.weights[i]);			
		}
		return clone;
	}
	
	public void cloneFrom(LogLinearWeights source) {
		for(int i=0; i<length; i++) {
				this.weights[i] = source.weights[i];
		}
	}
	
	public boolean equalsExact(LogLinearWeights other) {
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
	
	public boolean equalsApprox(LogLinearWeights other) {
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
