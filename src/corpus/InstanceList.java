package corpus;

import java.util.ArrayList;
import util.MathUtils;
import util.Timing;

public class InstanceList extends ArrayList<Instance> {
	private static final long serialVersionUID = -2409272084529539276L;
	int numberOfTokens;
	
	public InstanceList() {
		super();
	}

	public double getConditionalLogLikelihoodUsingPosteriorDistribution(
			double[][] parameterMatrix) {
		double cll = 0;
		Timing timing = new Timing();
		timing.start();
		for (int n = 0; n < this.size(); n++) {
			Instance i = get(n);
			cll += i.getConditionalLogLikelihoodUsingPosteriorDistribution(parameterMatrix);
		}
		System.out.println("CLL computation time : " + timing.stop());
		return cll;
	}

	public double[][] getGradient(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		//TODO: can further speed up partitionCache calculation (because for different state in the same timestep, Z's remain fixed)  
		double[][] partitionCache = new double[this.numberOfTokens][this.get(0).model.nrStates];
		int tokenIndex = 0;
		for(int n=0; n<this.size(); n++) {
			Instance instance = get(n);
			for(int t=0; t<instance.T; t++) {
				for(int state=0; state < instance.model.nrStates; state++) {
					double[] conditionalVector = instance.getConditionalVector(t, state);
					double normalizer = 0.0;
					for (int v = 0; v < parameterMatrix.length; v++) {
						double[] weightVector = parameterMatrix[v];
						normalizer += Math.exp(MathUtils.dot(weightVector, conditionalVector));
					}
					partitionCache[tokenIndex][state] = normalizer;
				}
				tokenIndex++;
			}
		}
		
		System.out.println("Partition Cache creation time : " + timing.stop());
		
		timing.start();
		double gradient[][] = new double[parameterMatrix.length][parameterMatrix[0].length];
		for (int i = 0; i < parameterMatrix.length; i++) { // all vocab length
			for (int j = 0; j < parameterMatrix[0].length; j++) {
				tokenIndex = 0;
				for (int n = 0; n < this.size(); n++) {
					Instance instance = get(n);
					for (int t = 0; t < instance.T; t++) {
						for (int state = 0; state < instance.model.nrStates; state++) {
							double posteriorProb = instance.posteriors[t][state];
							double[] conditionalVector = instance.getConditionalVector(t, state);
							if (i == instance.words[t][0]) {
								gradient[i][j] += posteriorProb * conditionalVector[j];
							}
							double normalizer = partitionCache[tokenIndex][state];									
							double numerator = Math.exp(MathUtils.dot(parameterMatrix[i], conditionalVector));
							gradient[i][j] -= posteriorProb * numerator / normalizer * conditionalVector[j];							
						}
						tokenIndex++;						
					}
				}
			}
		}
		
		System.out.println("Gradient computation time : " + timing.stop());		
		return gradient;
	}
}
