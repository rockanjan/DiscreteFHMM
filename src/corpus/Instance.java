package corpus;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeSet;

import javax.management.RuntimeErrorException;

import util.LogExp;
import util.MathUtils;
import util.SmoothWord;

import model.HMMBase;
import model.HMMType;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;
import model.inference.ForwardBackwardScaled;

public class Instance {
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public ForwardBackward forwardBackward;
	public int nrStates;
	public int unknownCount;
	public HMMBase model;
	
	//store the posteriors (important for training log-linear weights in M-step)
	public double[][] posteriors;
	
	public int[] decoded;

	public double[][] observationCache;

	public Instance(Corpus c, String line) {
		this.c = c;
		unknownCount = 0;
		// read from line
		populateWordArray(line);
	}
	
	public void doInference(HMMBase model) {
		this.model = model;
		// forwardBackward = new ForwardBackwardNoScaling(model, this);
		if (model.hmmType == HMMType.LOG_SCALE) {
			forwardBackward = new ForwardBackwardLog(model, this);
		} else {
//			System.out.println("ONLY LOG FORWARD BACKWARD IMPLEMENTED");
//			System.exit(-1);
			forwardBackward = new ForwardBackwardScaled(model, this);
		}
		nrStates = model.nrStates;
		forwardBackward.doInference();
	}

	public void clearInference() {
		forwardBackward.clear();
		forwardBackward = null;
		observationCache = null;
	}
	
	/*
	public void clearPosteriors() {
		posteriors = null;
	}
	*/
	
	/*
	 * returns log(ObservationProb)
	 */
	public double getObservationProbability(int position, int state) {
		if (observationCache == null) {
			observationCache = new double[T][nrStates];
			for (int t = 0; t < T; t++) {
				for(int s=0; s<nrStates; s++) {
					//TODO: store the log of the normalizer and also log of the numerator to avoid overflow
					double[] conditionalVector = getConditionalVector(t, s);
					
					int observationIndex = this.words[t][0];
					double[] weightObservation = model.param.weights.weights[observationIndex];
					double numerator = Math.exp(MathUtils.dot(conditionalVector, weightObservation));
					double result = numerator / getExactNormalizer(t, s, model.param.weights.weights);
					if(model.hmmType == HMMType.LOG_SCALE) {
						result = Math.log(result);
					}
					observationCache[t][s] = result;
				}				
			}
		}
		return observationCache[position][state];
	}
	
	
	public double[] getConditionalVector(int t, int state){
		double[] conditionalVector = new double[model.param.weights.conditionalSize];
		//fill the conditionVector
		conditionalVector[0] = 1.0; //offset
		int index = 1;
		for(int i=0; i<nrStates; i++) {
			if(i == state) {
				conditionalVector[index] = 1.0;
			} else {
				conditionalVector[index] = 0.0;
			}
			index++;
		}
		for(int z=1; z<model.corpus.oneTimeStepObsSize; z++) {
			for(int i=0; i<model.corpus.corpusVocab.get(z).vocabSize; i++) {
				if( this.words[t][z] == i) {
					conditionalVector[index] = 1.0;
				} else {
					conditionalVector[index] = 0.0;
				}
				index++;
			}
		}
		return conditionalVector;
	}
	
	/*
	 * gets conditional vector using viterbi decoded (state is fixed for a time t)
	 */
	public double[] getConditionalVectorUsingViterbiDecoded(int t){
		double[] conditionalVector = new double[model.param.weights.conditionalSize];
		//fill the conditionVector
		conditionalVector[0] = 1.0; //offset
		int index = 1;
		for(int i=0; i<nrStates; i++) {
			if(i == decoded[t]) {
				conditionalVector[index] = 1.0;
			} else {
				conditionalVector[index] = 0.0;
			}
			index++;
		}
		for(int z=1; z<c.oneTimeStepObsSize; z++) {
			for(int i=0; i<c.corpusVocab.get(z).vocabSize; i++) {
				if(words[t][z] == i) {
					conditionalVector[index] = 1.0;
				} else {
					conditionalVector[index] = 0.0;
				}
				index++;
			}
		}
		return conditionalVector;
	}
	
	
	public double getConditionalLogLikelihoodUsingViterbiDecoded(double[][] weights) {
		double cll = 0.0;
		for(int t=0; t<T; t++) {
			double normalizer = 0.0;
			double[] conditionalVector = getConditionalVectorUsingViterbiDecoded(t);
			for (int i=0; i<model.corpus.corpusVocab.get(0).vocabSize; i++) {
				double[] weightIthVocab = weights[i];
				normalizer += Math.exp(MathUtils.dot(conditionalVector, weightIthVocab));
			}
			int observationIndex = this.words[t][0];
			double[] weightObservation = weights[observationIndex];
			double numerator = Math.exp(MathUtils.dot(conditionalVector, weightObservation));
			double result = Math.log(numerator / normalizer);
			cll += result;
		}
		return cll;
	}
	
	public double getConditionalLogLikelihoodUsingPosteriorDistribution(double[][] weights) {
		double cll = 0.0;
		for(int t=0; t<T; t++) {
			for(int state=0; state<nrStates; state++) {
				double posteriorProb = posteriors[t][state];
				double normalizer = 0.0;
				double[] conditionalVector = getConditionalVector(t, state);
				//normalizer
				for (int i=0; i<model.corpus.corpusVocab.get(0).vocabSize; i++) {
					double[] weightIthVocab = weights[i];
					normalizer += Math.exp(MathUtils.dot(conditionalVector, weightIthVocab));
				}
				int observationIndex = this.words[t][0];
				double[] weightObservation = weights[observationIndex];
				double numerator = Math.exp(MathUtils.dot(conditionalVector, weightObservation));
				double result = posteriorProb * Math.log(numerator / normalizer); //expected CLL
				cll += result;
			}
		}
		return cll;
	}
	
	public double getApproxConditionalLogLikelihoodUsingPosteriorDistribution(double[][] weights) {
		double cll = 0.0;
		for(int t=0; t<T; t++) {
			for(int state=0; state<nrStates; state++) {
				double posteriorProb = posteriors[t][state];
				double normalizer = 0.0;
				double[] conditionalVector = getConditionalVector(t, state);
				//
				
				int observationIndex = this.words[t][0];
				double[] weightObservation = weights[observationIndex];
				double numerator = Math.exp(MathUtils.dot(conditionalVector, weightObservation));
				//double result = posteriorProb * Math.log(numerator / normalizer); //expected CLL
				double result = posteriorProb * Math.log(numerator / getApproxNormalizer(t, state, weights)); //expected CLL
				cll += result;
			}
		}
		return cll;
	}
		
	public void createDecodedViterbiCache(){
		decoded = new int[T];
		double[][] probLattice = new double[T][model.nrStates];
		int[][] stateLattice = new int[T][nrStates];
		
		for(int i=0; i<nrStates; i++) {
			double init = model.param.initial.get(0).get(i, 0);
			double obs = getObservationProbability(0, i);
			probLattice[0][i] = init + obs;			
		}
		
		double maxValue = -Double.MAX_VALUE;
		int maxIndex = -1;
		for(int t=1; t<T; t++) {
			for(int j=0; j<model.nrStates; j++) {
				//double obs = model.param.observation.get(instance.words[t], j);
				double obs = getObservationProbability(t, j);
				maxValue = -Double.MAX_VALUE;
				maxIndex = -1;
				for(int i=0; i<model.nrStates; i++) {
					double value = probLattice[t-1][i] + model.param.transition.get(0).get(j, i) + obs;
					if(value > maxValue) {
						maxValue = value;
						maxIndex = i;
					}
				}
				probLattice[t][j] = maxValue;
				stateLattice[t][j] = maxIndex;
			}
		}
		maxValue = -Double.MAX_VALUE;
		maxIndex = -1;
		for(int i=0; i<nrStates; i++) {
			if(probLattice[T-1][i] > maxValue) {
				maxValue = probLattice[T-1][i];
				decoded[T-1] = i;
			}
		}
		//backtrack
		for(int t=T-2; t>=0; t--) {
			decoded[t] = stateLattice[t+1][decoded[t+1]];			
		}				
	}
	
	public double getApproxNormalizer(int position, int state, double[][] weights) {
		double Z = 0;
		double[] conditionalVector = getConditionalVector(position, state);
		
		TreeSet<Integer> randomVocabSet = Corpus.getRandomVocabSet();
		int currentToken = words[position][0];
		randomVocabSet.add(currentToken);
		
		for(int randomV : randomVocabSet) {
			double numerator = Math.exp(MathUtils.dot(weights[randomV], conditionalVector));
			Z += numerator;
		}
		Z = Z * model.corpus.corpusVocab.get(0).vocabSize / (randomVocabSet.size());
		return Z;
	}
	
	public double getExactNormalizer(int position, int state, double[][] weights) {
		double Z = 0;
		double[] conditionalVector = getConditionalVector(position, state);
		for(int i=0; i<model.corpus.corpusVocab.get(0).vocabSize; i++) {
			//double numerator = Math.exp(MathUtils.dot(weights[i], conditionalVector));
			double numerator = LogExp.expApprox(MathUtils.dot(weights[i], conditionalVector));
			Z += numerator;
		}
		return Z;
	}

	/*
	 * returns the original word at the position
	 */
	public String getWord(int position) {
		return c.corpusVocab.get(0).indexToWord.get(words[position][0]);
	}

	public void populateWordArray(String line) {
		String allTimeSteps[] = line.split(c.delimiter);
		T = allTimeSteps.length;
		words = new int[T][c.oneTimeStepObsSize];
		for (int i = 0; i < T; i++) {
			String oneTimeStep = allTimeSteps[i];
			String[] obsElements = oneTimeStep.split(c.obsDelimiter);
			if(obsElements.length != c.oneTimeStepObsSize) {
				throw new RuntimeException("One timestep observation size from vocab : " + c.oneTimeStepObsSize + " from instance: " + obsElements.length);
			}
			// original word
			String word = obsElements[0];
			if (c.corpusVocab.get(0).lower) {
				word = word.toLowerCase();
			}
			if (c.corpusVocab.get(0).smooth) {
				word = SmoothWord.smooth(word);
			}
			int wordId = c.corpusVocab.get(0).getIndex(word);
			words[i][0] = wordId;
			// for hmm states as observations
			for (int j = 1; j < obsElements.length; j++) {
				String obsElement = obsElements[j];
				int obsElementId = c.corpusVocab.get(j).getIndex(obsElement);
				words[i][j] = obsElementId;
			}
		}
	}
}
