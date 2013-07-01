package corpus;

import program.Main;
import model.HMMBase;
import model.HMMType;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;
import model.inference.ForwardBackwardScaled;
import model.param.LogLinearWeights;
import util.MathUtils;
import util.SmoothWord;

public class Instance {
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public ForwardBackward forwardBackward;
	//TODO: might change if we consider finalState hmm
	public static int nrStates = Main.numStates;
	public int unknownCount;
	public HMMBase model;
	
	public static int FEATURE_PARTITION_CACHE_SIZE = 10000;
	
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
		if (model.hmmType == HMMType.LOG_SCALE) {
			forwardBackward = new ForwardBackwardLog(model, this);
		} else {
			forwardBackward = new ForwardBackwardScaled(model, this);
		}
		//nrStates = model.nrStates;
		forwardBackward.doInference();
	}

	public void clearInference() {
		forwardBackward.clear();
		forwardBackward = null;
		observationCache = null;
	}
	
	/*
	 * returns log(ObservationProb) or ObservationProb depending on the model
	 */
	public double getObservationProbability(int position, int state) {
		if (observationCache == null) {
			observationCache = new double[T][nrStates];
			for (int t = 0; t < T; t++) {
				for(int s=0; s<nrStates; s++) {
					double[] conditionalVector = getConditionalVector(t, s);
					int observationIndex = this.words[t][0];
					double[] expWeightObservation = model.param.expWeightsCache[observationIndex];
					double numerator = MathUtils.expDot(expWeightObservation, conditionalVector);
					double result = numerator / getExactNormalizer(t, s, model.param.expWeightsCache);
					if(model.hmmType == HMMType.LOG_SCALE) {
						result = Math.log(result);
					}
					observationCache[t][s] = result;
				}				
			}
		}
		return observationCache[position][state];
	}
	
	public String getConditionalString(int t, int state) {
		StringBuffer sb = new StringBuffer();
		sb.append(state);
		for(int z=1; z<Corpus.oneTimeStepObsSize; z++) {
			sb.append("+" + this.words[t][z]);
		}
		return sb.toString();
	}
	
	
	public double[] getConditionalVector(int t, int state){
		//double[] conditionalVector = new double[model.param.weights.conditionalSize];
		double[] conditionalVector = new double[LogLinearWeights.conditionalSize];
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
		for(int z=1; z<Corpus.oneTimeStepObsSize; z++) {
			for(int i=0; i<Corpus.corpusVocab.get(z).vocabSize; i++) {
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
		for(int z=1; z<Corpus.oneTimeStepObsSize; z++) {
			for(int i=0; i<Corpus.corpusVocab.get(z).vocabSize; i++) {
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
	
	public double getConditionalLogLikelihoodUsingPosteriorDistribution(double[][] expWeights) {
		double cll = 0.0;
		for(int t=0; t<T; t++) {
			for(int state=0; state<nrStates; state++) {
				double posteriorProb = posteriors[t][state];
				double normalizer = getExactNormalizer(t, state, expWeights);
				int observationIndex = this.words[t][0];
				double[] conditionalVector = getConditionalVector(t, state);
				//double numerator = MathUtils.exp(MathUtils.dot(conditionalVector, weightObservation));
				double numerator = MathUtils.expDot(expWeights[observationIndex], conditionalVector);
				double result = posteriorProb * Math.log(numerator / normalizer); //expected CLL
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
	
	public double getExactNormalizer(int position, int state, double[][] expWeights) {
		double Z = 0;
		double[] conditionalVector = getConditionalVector(position, state);
		String conditionalString = getConditionalString(position, state);
		if(InstanceList.featurePartitionCache.containsKey(conditionalString)) {
			return InstanceList.featurePartitionCache.get(conditionalString);
		}
		for(int i=0; i<Corpus.corpusVocab.get(0).vocabSize; i++) {
			double numerator = MathUtils.expDot(expWeights[i], conditionalVector);
			Z += numerator;
		}
		if(InstanceList.featurePartitionCache.size() < FEATURE_PARTITION_CACHE_SIZE) {
			InstanceList.featurePartitionCache.put(conditionalString, Z);
		}		
		return Z;
	}

	/*
	 * returns the original word at the position
	 */
	public String getWord(int position) {
		return Corpus.corpusVocab.get(0).indexToWord.get(words[position][0]);
	}

	public void populateWordArray(String line) {
		String allTimeSteps[] = line.split(Corpus.delimiter);
		T = allTimeSteps.length;
		words = new int[T][Corpus.oneTimeStepObsSize];
		for (int i = 0; i < T; i++) {
			String oneTimeStep = allTimeSteps[i];
			String[] obsElements = oneTimeStep.split(Corpus.obsDelimiter);
			if(obsElements.length != Corpus.oneTimeStepObsSize) {
				throw new RuntimeException("One timestep observation size from vocab : " + Corpus.oneTimeStepObsSize + " from instance: " + obsElements.length);
			}
			// original word
			String word = obsElements[0];
			if (Corpus.corpusVocab.get(0).lower) {
				word = word.toLowerCase();
			}
			if (Corpus.corpusVocab.get(0).smooth) {
				word = SmoothWord.smooth(word);
			}
			int wordId = Corpus.corpusVocab.get(0).getIndex(word);
			if(wordId == 0) {
				unknownCount++;
			}
			words[i][0] = wordId;
			// for hmm states as observations
			for (int j = 1; j < obsElements.length; j++) {
				String obsElement = obsElements[j];
				int obsElementId = Corpus.corpusVocab.get(j).getIndex(obsElement);
				words[i][j] = obsElementId;
			}
		}
	}
}
