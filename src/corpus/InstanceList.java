package corpus;

import java.util.ArrayList;

import util.MathUtils;
import util.MyArray;
import util.Timing;

public class InstanceList extends ArrayList<Instance> {
	private static final long serialVersionUID = -2409272084529539276L;

	public InstanceList() {
		super();
	}

	public double getConditionalLogLikelihoodUsingPosteriorDistribution(
			double[][] parameterMatrix) {
		double cll = 0;
		for (int n = 0; n < this.size(); n++) {
			Instance i = get(n);
			cll += i.getConditionalLogLikelihoodUsingPosteriorDistribution(parameterMatrix);
		}
		return cll;
	}

	public double[][] getGradient(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		double gradient[][] = new double[parameterMatrix.length][parameterMatrix[0].length];
		for (int i = 0; i < parameterMatrix.length; i++) { // all vocab length
			for (int j = 0; j < parameterMatrix[0].length; j++) {
				for (int n = 0; n < this.size(); n++) {
					Instance instance = get(n);
					for (int t = 0; t < instance.T; t++) {
						for (int state = 0; state < instance.model.nrStates; state++) {
							double posteriorProb = instance.posteriors[t][state];
							double[] conditionalVector = instance
									.getConditionalVector(t, state);
							if (i == instance.words[t][0]) {
								gradient[i][j] += conditionalVector[j];
							}
							// expected value subtraction
							double normalizer = 0.0;
							for (int v = 0; v < parameterMatrix.length; v++) { // all
																				// vocabs
								double[] weightVector = parameterMatrix[v];
								normalizer += Math.exp(MathUtils.dot(
										weightVector, conditionalVector));
							}
							double numerator = Math.exp(MathUtils.dot(
									parameterMatrix[i], conditionalVector));
							gradient[i][j] -= numerator / normalizer * conditionalVector[j];
							gradient[i][j] = posteriorProb * gradient[i][j];
						}						
					}
				}
			}
		}
		/*
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < parameterMatrix.length; i++) { // all vocab length
			for (int j = 0; j < parameterMatrix[0].length; j++) {
				if (gradient[i][j] < min) {
					min = gradient[i][j];
				}
				if (gradient[i][j] > max) {
					max = gradient[i][j];
				}
			}
		}
		System.out.format("Gradient: min %.3f max %.3f\n", min, max);
		System.out.println("Gradient computation time : " + timing.stop());
		*/
		return gradient;
		// return MyArray.createVector(gradient);
	}
}
