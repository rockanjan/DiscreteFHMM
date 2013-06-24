package model.train;

import util.MathUtils;
import util.MyArray;
import util.Timing;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class SgdTrainer {
	Corpus corpus;
	public static double adaptiveStep;
	double eta = 10;
	double t0 = 100;
	double precision = 1e-2;

	public SgdTrainer(Corpus corpus) {
		this.corpus = corpus;
	}

	public void train(double[][] parameterMatrix, int maxIter) {
		//double[][] newWeights = MyArray.getCloneOfMatrix(parameterMatrix);
		Timing timing = new Timing();
		InstanceList instanceList = corpus.trainInstanceMStepSampleList;
		for(int i=0; i<maxIter; i++) {
			double[][] oldWeights = MyArray.getCloneOfMatrix(parameterMatrix);
			
			double[][] gradient = instanceList.getGradient(parameterMatrix);
			MathUtils.matrixElementWiseMultiplication(gradient, adaptiveStep);
			MathUtils.addMatrix(parameterMatrix, gradient);
			System.out.println("CLL : " + instanceList.getConditionalLogLikelihoodUsingPosteriorDistribution(parameterMatrix));
			double differenceNorm = MathUtils.matrixDifferenceNorm(oldWeights, parameterMatrix);
			System.out.println("Sgd Diff: " + differenceNorm);
			if(differenceNorm < precision) {
				System.out.println("Sgd converged " + differenceNorm);
				break;
			}
		}
		System.out.println("Sgd Training time : " + timing.stop());
	}
}
