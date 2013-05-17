package model.train;

import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

public class LogLinearWeightsOptimizable implements Optimizable.ByGradientValue{
	
	double[] parameters;
	boolean paramChanged = true;
	double latestValue = 0.0;
	double[] latestGradient;
	Corpus corpus;
	
	double c2 = 0.5; //regularizer
	
	public LogLinearWeightsOptimizable(double[] initParams, Corpus corpus) {
		this.corpus = corpus;
		parameters = new double[initParams.length];
		for(int i=0; i<initParams.length; i++) {
			parameters[i] = initParams[i];
		}
		latestGradient = new double[parameters.length];
	}
	
	/*
	 * returns the Conditional Log likelihood of the training corpus
	 * 
	 */
	@Override
	public double getValue() {
		if(paramChanged) {
			double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
			double cll = corpus.trainInstanceList.getConditionalLogLikelihood(weights);
			//add regularizer
			double normSquared = MyArray.getL2NormSquared(parameters);
			latestValue = cll - c2 * normSquared;
			
		}
        return latestValue;
	}

	@Override
	public void getValueGradient(double[] gradient) {
		if(paramChanged) {
			double[][] weights = MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
			double[][] newGradients = corpus.trainInstanceList.getGradient(weights);
			//regularizer
			for(int i=0; i<newGradients.length; i++) {
				for(int j=0; j<newGradients[0].length; j++) {
					newGradients[i][j] -= 2 * c2 *  weights[i][j];
				}
			}
			double[] newGradientsVectorized = MyArray.createVector(newGradients);
			weights = null;
			newGradients = null;
	        for(int i=0; i<parameters.length; i++) {
				latestGradient[i] = newGradientsVectorized[i];
			}
	        newGradientsVectorized = null;
		} else {
			for(int i=0; i<parameters.length; i++) {
				gradient[i] = latestGradient[i];
			}
		}
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
	
	public double[][] getParameterMatrix() {
		return MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
	}

	@Override
	public void setParameter(int i, double value) {
		System.out.println("set parameter called");
		parameters[i] = value;
		paramChanged = true;
	}

	@Override
	public void setParameters(double[] newParam) {
		System.out.println("set parameters called");
		for(int i=0; i<parameters.length; i++) {
			parameters[i] = newParam[i];
		}
		paramChanged = true;
	}
}
