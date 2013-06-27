package model.train;

import util.MathUtils;
import util.MyArray;
import util.Timing;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class SgdTrainer {
	Corpus corpus;
	public static double adaptiveStep=1;
	double eta = 10;
	double t0 = 100;
	double precision = 1e-4;

	public SgdTrainer(Corpus corpus) {
		this.corpus = corpus;
	}

	//TODO consider regularization
	public void train(double[][] parameterMatrix, int maxIter) {
		Timing timing = new Timing();
		InstanceList instanceList = corpus.trainInstanceMStepSampleList;
		for(int i=0; i<maxIter; i++) {
			double oldCLL = instanceList.getConditionalLogLikelihoodUsingPosteriorDistribution(parameterMatrix);
			double[][] gradient = instanceList.getGradient(parameterMatrix);
			double[] stepSizes = new double[10];
			//set step sizes for backtracking
			stepSizes[0] = adaptiveStep; //previously found best adaptiveStep
			for(int j=1; j<stepSizes.length; j++) {
				stepSizes[j] = stepSizes[j-1] * 0.8; //decrease geometrically 
			}
			MyArray.printVector(stepSizes, "Step sizes");
			//find the best step
			double maxValue = -Double.MAX_VALUE;
			for(double s : stepSizes) {
				double[][] newParamMatrix = MyArray.getCloneOfMatrix(parameterMatrix);
				double[][] scaledGradient = MyArray.getCloneOfMatrix(gradient);
				MathUtils.matrixElementWiseMultiplication(scaledGradient, s);
				MathUtils.addMatrix(newParamMatrix, scaledGradient);
				double CLL = instanceList.getConditionalLogLikelihoodUsingPosteriorDistribution(newParamMatrix);
				if(CLL > maxValue) {
					maxValue = CLL;
					adaptiveStep = s;
				}
			}
			System.out.println("CLL in sgd : " + maxValue);
			MathUtils.matrixElementWiseMultiplication(gradient, adaptiveStep);
			MathUtils.addMatrix(parameterMatrix, gradient);
			double CLL = instanceList.getConditionalLogLikelihoodUsingPosteriorDistribution(parameterMatrix);
			if( Math.abs(1- oldCLL/CLL) < precision ) {
				System.out.println("sgd converged.");
				break;
			}
		}
		System.out.println("Sgd Training time : " + timing.stop());
	}
}
