package model.train;

import config.Config;
import util.MyArray;
import cc.mallet.optimize.Optimizable;
import corpus.Corpus;

public class CLLTrainerJoint implements Optimizable.ByGradientValue{
	
	double[] parameters;
	double latestValue = 0.0;
	Corpus corpus;
	
	public int gradientCallCount = 0;
	
	public CLLTrainerJoint(double[] initParams, Corpus corpus) {
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
		double cll = corpus.trainInstanceMStepSampleList.getCLLJoint(parameters);
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
		double[] newGradients = corpus.trainInstanceMStepSampleList.getJointGradient(parameters);
		//regularizer
		for(int i=0; i<newGradients.length; i++) {
			gradient[i] = newGradients[i] - 2 * Config.c2 *  newGradients[i];
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
		return MyArray.createMatrix(parameters, corpus.corpusVocab.get(0).vocabSize);
	}
	
	public double[] getParameterVector() {
		return parameters.clone();
	}

	
	/************ Debugging code *********/
	
	private double[] getFiniteDifferenceGradient() {
		double[] newGradients = new double[parameters.length];
		double step = 1e-8;
		double[] weights = parameters.clone();
		for(int i=0; i<parameters.length; i++) {
			weights[i] = weights[i] - step;
			double valueX = corpus.trainInstanceEStepSampleList.getCLLJoint(weights);
			weights[i] = weights[i] + step + step;
			double valueXStepped = corpus.trainInstanceEStepSampleList.getCLLJoint(weights);
			newGradients[i] =  valueXStepped/ (2*step) - valueX / (2*step);
			//System.out.println("grad from finitedifference = " + newGradients[i][j]);
			//reset weights
			weights[i] = weights[i] - step;
		}
		return newGradients;
	}
	
	private double[] getGradientByEquation() {
		double[] weights = parameters.clone();
		double[] newGradients = corpus.trainInstanceEStepSampleList.getJointGradient(weights);
		return newGradients;
	}
	
	public void checkGradientComputation() {
		System.out.println("Checking joint gradient computation");
		double[] finiteDifferenceGradient = getFiniteDifferenceGradient();
		double[] equationGradient = getGradientByEquation();
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
