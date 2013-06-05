package corpus;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import cc.mallet.util.CommandOption.Set;

import util.MathUtils;
import util.Timing;

public class InstanceList extends ArrayList<Instance> {
	private static final long serialVersionUID = -2409272084529539276L;
	public int numberOfTokens;
	
	int VOCAB_SAMPLE_SIZE = 200;
	Random random = new Random(37);
	
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
	
	public double[][] getGradientModified(double[][] parameterMatrix) {
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
		tokenIndex = 0;
		for (int n = 0; n < this.size(); n++) {
			Instance instance = get(n);
			for (int t = 0; t < instance.T; t++) {
				for(int j=0; j<parameterMatrix[0].length; j++) {
					for (int state = 0; state < instance.model.nrStates; state++) {
						double posteriorProb = instance.posteriors[t][state];
						double[] conditionalVector = instance.getConditionalVector(t, state);
						gradient[instance.words[t][0]][j] += posteriorProb * conditionalVector[j];
						double normalizer = partitionCache[tokenIndex][state];								
						for(int v=0; v<parameterMatrix.length; v++) {
							double numerator = Math.exp(MathUtils.dot(parameterMatrix[v], conditionalVector));
							gradient[v][j] -= posteriorProb * numerator / normalizer * conditionalVector[j];
						}							
					}
				}
				tokenIndex++;						
			}
		}
		System.out.println("Gradient computation time : " + timing.stop());		
		return gradient;
	}
	
	public double[][] getGradientApprox(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		int tokenIndex = 0;

		timing.start();
		double gradient[][] = new double[parameterMatrix.length][parameterMatrix[0].length];
		tokenIndex = 0;
		for (int n = 0; n < this.size(); n++) {
			Instance instance = get(n);
			for (int t = 0; t < instance.T; t++) {
				for(int j=0; j<parameterMatrix[0].length; j++) {
					for (int state = 0; state < instance.model.nrStates; state++) {
						double posteriorProb = instance.posteriors[t][state];
						double[] conditionalVector = instance.getConditionalVector(t, state);
						gradient[instance.words[t][0]][j] += posteriorProb * conditionalVector[j];
						
						//TODO: efficient sampling
						double[] sampleNumerator = new double[parameterMatrix.length];
						double Z = 0;
						HashSet<Integer> sampled = new HashSet<Integer>();
						for(int i=0; i<VOCAB_SAMPLE_SIZE; i++) {
							int randomV = random.nextInt(instance.model.corpus.corpusVocab.get(0).vocabSize);
							sampled.add(randomV);
							double numerator = Math.exp(MathUtils.dot(parameterMatrix[randomV], conditionalVector));
							sampleNumerator[randomV] += numerator; // addition because we might have repeated sampling
							Z += numerator;
						}
						
						//rescale Z
						Z = Z * instance.model.corpus.corpusVocab.get(0).vocabSize / VOCAB_SAMPLE_SIZE;
						for(Integer v : sampled) {
							gradient[v][j] -= posteriorProb * sampleNumerator[v] / Z * conditionalVector[j];
						}							
					}
				}
				tokenIndex++;						
			}
		}
		System.out.println("Gradient computation time : " + timing.stop());		
		return gradient;
	}
}
