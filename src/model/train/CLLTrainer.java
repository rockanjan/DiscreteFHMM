package model.train;

import config.Config;
import util.MyArray;
import cc.mallet.optimize.Optimizable;
import corpus.Corpus;

public class CLLTrainer implements Optimizable.ByGradientValue{
	
	double[] parameters;
	double latestValue = 0.0;
	Corpus corpus;
	
	public int gradientCallCount = 0;
	
	public CLLTrainer(double[] initParams, Corpus corpus) {
		this.corpus = corpus;
		parameters = new double[initParams.length];
		for(int i=0; i<initParams.length; i++) {
			parameters[i] = initParams[i];
		}
	}
	
	/*
	 * returns the Conditional Log likelihood of the training corpus
	 * 
	 */
	@Override
	public double getValue() {
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		double cll = corpus.trainInstanceMStepSampleList.getCLL(weights);
		//add regularizer
		double normSquared = MyArray.getL2NormSquared(parameters);
		latestValue = cll - Config.c2 * normSquared;
		if(Config.displayDetail) {
			System.out.println("CLL : " + latestValue);
		}
        return latestValue;
	}

	@Override
	public void getValueGradient(double[] gradient) {
		gradientCallCount++;
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		//double[][] newGradients = corpus.trainInstanceMStepSampleList.getGradient(weights);
		double[][] newGradients = corpus.trainInstanceMStepSampleList.getGradient(weights);
		//regularizer
		for(int i=0; i<newGradients.length; i++) {
			for(int j=0; j<newGradients[0].length; j++) {
				newGradients[i][j] -= 2 * Config.c2 *  weights[i][j];
			}
		}
		weights = null;
		double[] newGradientsVectorized = MyArray.createVector(newGradients);
		newGradients = null;
		for(int i=0; i<parameters.length; i++) {
			gradient[i] = newGradientsVectorized[i];
		}
		newGradientsVectorized = null;        
	}
	
	@Override
	public int getNumParameters() {
		return parameters.length;
	}

	@Override
	public double getParameter(int i) {
		return parameters[i];
	}

	@Override
	public void getParameters(double[] buffer) {
		for(int i=0; i<parameters.length; i++) {
			buffer[i] = parameters[i];
		}
		
	}
	
	@Override
	public void setParameter(int i, double value) {
		//System.out.println("set parameter called");
		parameters[i] = value;
	}

	@Override
	public void setParameters(double[] newParam) {
		//System.out.println("set parameters called");
		for(int i=0; i<parameters.length; i++) {
			parameters[i] = newParam[i];
		}
	}
	
	public double[][] getParameterMatrix() {
		return MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
	}

	
	/************ Debugging code *********/
	
	private double[][] getFiniteDifferenceGradient() {
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		double[][] newGradients = new double[weights.length][weights[0].length];
		double step = 1e-8;
		for(int i=0; i<weights.length; i++) {
			for(int j=0; j<weights[0].length; j++) {
				weights[i][j] = weights[i][j] - step;
				double valueX = corpus.trainInstanceEStepSampleList.getCLL(weights);
				weights[i][j] = weights[i][j] + step + step;
				double valueXStepped = corpus.trainInstanceEStepSampleList.getCLL(weights);
				newGradients[i][j] =  valueXStepped/ (2*step) - valueX / (2*step);
				//System.out.println("grad from finitedifference = " + newGradients[i][j]);
				//reset weights
				weights[i][j] = weights[i][j] - step;
			}
		}
		return newGradients;
	}
	
	private double[][] getGradientByEquation() {
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		double[][] newGradients = corpus.trainInstanceEStepSampleList.getGradient(weights);
		return newGradients;
	}
	
	public void checkGradientComputation() {
		double[] finiteDifferenceGradient = MyArray.createVector(getFiniteDifferenceGradient());
		double[] equationGradient = MyArray.createVector(getGradientByEquation());
		double[] difference = new double[finiteDifferenceGradient.length];
		double maxDiff = -Double.MAX_VALUE;
		double minDiff = Double.MAX_VALUE;
		for(int i=0; i<finiteDifferenceGradient.length; i++) {
			difference[i] = finiteDifferenceGradient[i] - equationGradient[i];
			if(difference[i] > maxDiff) {
				maxDiff = difference[i];
			}
			if(difference[i] < minDiff) {
				minDiff = difference[i];
			}
			//System.out.format("%.9f, %.9f, diff=%.9f \n", finiteDifferenceGradient[i], equationGradient[i], difference[i]);
		}
		System.out.format("Gradient Difference: Max %.5f, Min %.5f\n", maxDiff, minDiff);
	}
	/*************** Debugging code *********/
}
