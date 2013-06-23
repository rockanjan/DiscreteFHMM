package model.train;

import util.MathUtils;
import util.MyArray;
import util.Timing;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class AveragedPerceptronTrainerPosterior {
	Corpus corpus;
	public static double adaptiveStep;
	double eta = 10;
	double t0 = 100;
	double precision = 1e-2;
	public AveragedPerceptronTrainerPosterior(Corpus corpus) {
		this.corpus = corpus;
	}

	public void train(double[][] parameterMatrix, int maxIter) {
		Timing timing = new Timing();
		InstanceList instanceList = corpus.trainInstanceMStepSampleList;		
		double[][] averagedWeight = new double[parameterMatrix.length][parameterMatrix[0].length];
		for(int i=0; i<maxIter; i++) {
			double[][] oldWeights = MyArray.getCloneOfMatrix(parameterMatrix);
			for (int n = 0; n < instanceList.size(); n++) {
				Instance instance = instanceList.get(n);
				for (int t = 0; t < instance.T; t++) {
					for (int state = 0; state < instance.model.nrStates; state++) {
						double posteriorProb = instance.posteriors[t][state];
						double[] conditionalVector = instance.getConditionalVector(t, state);
						//perceptron training collins: max-margin
						//find the argmax vocab predicted by the model
						int maxIndex = -1;
						double maxProb = -Double.MAX_VALUE;
						for(int v = 0; v<parameterMatrix.length; v++) { //all vocabs
							double prob = MathUtils.dot(conditionalVector, parameterMatrix[v]);
							if(prob > maxProb) {
								maxIndex = v;
								maxProb = prob;
							}
						}
						if(maxIndex != instance.words[t][0]) {
							for(int j=0; j<conditionalVector.length; j++) {
								parameterMatrix[instance.words[t][0]][j] += adaptiveStep * posteriorProb * conditionalVector[j];
								//incorrect word's feature value
								parameterMatrix[maxIndex][j] -= adaptiveStep * posteriorProb * conditionalVector[j];
							}
						}
					}
				}
				MathUtils.addMatrix(averagedWeight, parameterMatrix);
			}
			double differenceNorm = MathUtils.matrixDifferenceNorm(oldWeights, parameterMatrix);
			//System.out.println("Diff: " + differenceNorm);
			/*
			if(differenceNorm < precision) {
				System.out.println("Perceptron converged " + differenceNorm);
				break;
			}
			*/
		}
		int N = instanceList.numberOfTokens;
		double total = 1.0 * maxIter * N;
		MathUtils.matrixElementWiseMultiplication(averagedWeight, 1.0/total);
		System.out.println("Distance Norm : " + 
				MathUtils.matrixDifferenceNorm(parameterMatrix, averagedWeight));
		parameterMatrix = averagedWeight;
		System.out.println("Perceptron training time: " + timing.stop());
	}	
}