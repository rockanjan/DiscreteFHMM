package model.train;

import util.MathUtils;
import util.MyArray;
import util.Timing;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class PerceptronTrainer {
	Corpus corpus;
	double alpha = 0.1;
	double precision = 1e-3;
	public PerceptronTrainer(Corpus corpus) {
		this.corpus = corpus;
	}

	public void train(double[][] parameterMatrix, int maxIter) {
		Timing timing = new Timing();
		InstanceList instanceList = corpus.trainInstanceMStepSampleList;
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
								parameterMatrix[instance.words[t][0]][j] += alpha * posteriorProb * conditionalVector[j];
								//incorrect word's feature value
								parameterMatrix[maxIndex][j] -= alpha * posteriorProb * conditionalVector[j];
							}
						}
					}
				}
			}
			double differenceNorm = MathUtils.matrixDifferenceNorm(oldWeights, parameterMatrix);
			System.out.println("norm: " + differenceNorm);
			if(differenceNorm < precision) {
				System.out.println("Perceptron converged " + differenceNorm);
				break;
			}
		}
		System.out.println("Perceptron training time: " + timing.stop());
	}	
}