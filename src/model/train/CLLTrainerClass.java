package model.train;

import config.Config;
import util.MyArray;
import cc.mallet.optimize.Optimizable;
import corpus.Corpus;
import corpus.WordClass;

public class CLLTrainerClass implements Optimizable.ByGradientValue{
	
	double[] parameters;
	double latestValue = 0.0;
	Corpus corpus;
	
	public int gradientCallCount = 0;
	
	public CLLTrainerClass(double[] initParams, Corpus corpus) {
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
		double cll = corpus.trainInstanceMStepSampleList.getCLLClass(parameters);
		double normSquared = MyArray.getL2NormSquared(parameters);
		latestValue = cll - Config.c2 * normSquared;
		if(Config.displayDetail) {
			System.out.println("CLL Class : " + latestValue);
		}
        return latestValue;
	}

	@Override
	public void getValueGradient(double[] gradient) {
		gradientCallCount++;
		double[] newGradients = corpus.trainInstanceMStepSampleList.getGradientClass(parameters);
		//regularizer
		for(int i=0; i<newGradients.length; i++) {
			gradient[i] = newGradients[i] - 2 * Config.c2 *  parameters[i];
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
		return MyArray.createMatrix(parameters, WordClass.numClusters);
	}	
}
