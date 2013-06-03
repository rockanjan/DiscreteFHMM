package model.train;

import util.MathUtils;
import util.Timing;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class PerceptronTrainer {
	Corpus corpus;

	public PerceptronTrainer(Corpus corpus) {
		this.corpus = corpus;
	}

	public void train(double[][] parameterMatrix) {
		//double[][] newWeights = MyArray.getCloneOfMatrix(parameterMatrix);
		Timing timing = new Timing();
		InstanceList instanceList = corpus.trainInstanceMStepSampleList;

		for (int n = 0; n < instanceList.size(); n++) {
			Instance instance = instanceList.get(n);
			for (int t = 0; t < instance.T; t++) {
				for (int state = 0; state < instance.model.nrStates; state++) {
					double posteriorProb = instance.posteriors[t][state];
					double[] conditionalVector = instance.getConditionalVector(t, state);
					
					for(int v = 0; v<parameterMatrix.length; v++) { //all vocabs
						double[] weight_v = parameterMatrix[v];
						if(v == instance.words[t][0]) {
							if(MathUtils.dot(conditionalVector, weight_v) < 1) {
								updateWeight(posteriorProb, weight_v, conditionalVector);
							}
						}
						else {
							if(MathUtils.dot(conditionalVector, weight_v) > -1) {
								updateWeight(-posteriorProb, weight_v, conditionalVector);
							}
						}
					}
				}
			}
		}
	}
	
	public void updateWeight(double scale, double[] weight, double[] x) {
		for(int i=0; i<weight.length; i++) {
			weight[i] += scale * x[i];
		}
	}
}