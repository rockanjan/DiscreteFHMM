package model.train;

import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

public class SgdTrainer{
	
	double[] parameters;
	double latestValue = 0.0;
	Corpus corpus;
	
	public int gradientCallCount = 0;
	
	double c2 = 0.01; //regularizer
	
	public SgdTrainer(double[] initParams, Corpus corpus) {
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
	public double getValue() {
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		double cll = corpus.trainInstanceMStepSampleList.getConditionalLogLikelihoodUsingViterbi(weights);
		//double cll = corpus.trainInstanceMStepSampleList.getApproxConditionalLogLikelihoodUsingPosteriorDistribution(weights);
		//add regularizer
		double normSquared = MyArray.getL2NormSquared(parameters);
		latestValue = cll - c2 * normSquared;
		System.out.println("CLL : " + latestValue);
        return latestValue;
	}

	public void getValueGradient(double[] gradient) {
		gradientCallCount++;
		double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
		double[][] newGradients = corpus.trainInstanceMStepSampleList.getGradient(weights);
		//regularizer
		for(int i=0; i<newGradients.length; i++) {
			for(int j=0; j<newGradients[0].length; j++) {
				newGradients[i][j] -= 2 * c2 *  weights[i][j];
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
	
	public int getNumParameters() {
		return parameters.length;
	}

	public double getParameter(int i) {
		return parameters[i];
	}

	public void getParameters(double[] buffer) {
		for(int i=0; i<parameters.length; i++) {
			buffer[i] = parameters[i];
		}
		
	}
	
	public void setParameter(int i, double value) {
		//System.out.println("set parameter called");
		parameters[i] = value;
	}

	public void setParameters(double[] newParam) {
		//System.out.println("set parameters called");
		for(int i=0; i<parameters.length; i++) {
			parameters[i] = newParam[i];
		}
	}
	
	public double[][] getParameterMatrix() {
		return MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
	}
	
	public void train(int numIter) {
		int maxIter = 100; //max
		if(numIter > 0) maxIter = numIter;
		for(int iterCount=0; iterCount<maxIter; iterCount++) {
			
		}
	}
}
