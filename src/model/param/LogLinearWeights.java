package model.param;

import java.util.Random;

public class LogLinearWeights {
	public double[][] weights; //weights for the log-linear model
	
	public int vocabSize;
	public int conditionalSize;
	
	/*
	 * @param vocabSize = number of distinct observation tokens
	 * @param conditionSize = dimension of current hidden layer, |X|  
	 * and sum of dimensions of previous decodes states |Z_vector|
	 */
	public LogLinearWeights(int vocabSize, int xzSize) {
		this.vocabSize = vocabSize;
		this.conditionalSize = xzSize + 1;		 //1 for offset
	}
	public void initializeZeros() {
		weights = new double[vocabSize][conditionalSize]; 
	}
	
	public void initializeRandom(Random r) {
		weights = new double[vocabSize][conditionalSize];
		for(int y=0; y<vocabSize; y++) {
			for(int u=0; u<conditionalSize; u++) {
				weights[y][u] = r.nextGaussian();
			}
		}
	}
	
	public void cloneFrom(LogLinearWeights source) {
		for(int i=0; i<vocabSize; i++) {
			for(int j=0; j<conditionalSize; j++) {
				this.weights[i][j] = source.weights[i][j];
			}
		}
	}
}
