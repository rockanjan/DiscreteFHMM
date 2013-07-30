package corpus;

import java.util.ArrayList;
import java.util.List;

import program.Main;
import model.HMMBase;
import model.HMMType;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;
import model.inference.VariationalParam;
import model.inference.VariationalParamObservation;
import model.param.HMMParamBase;
import model.param.LogLinearWeights;
import util.MathUtils;
import util.MyArray;
import util.SmoothWord;

public class Instance {
	public VariationalParamObservation varParamObs;
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public List<ForwardBackward> forwardBackwardList;
	
	//TODO: might change if we consider finalState hmm
	public static int nrStates = Main.numStates;
	public int unknownCount;
	public HMMBase model;
	
	public static int FEATURE_PARTITION_CACHE_SIZE = 10000;
	
	//m,t,k
	public double[][][] posteriors;
	
	//m, t
	public int[][] decodedStates;

	public double[] observationCache;
	
	public double logLikelihood;

	public Instance(Corpus c, String line) {
		this.c = c;
		unknownCount = 0;
		// read from line
		populateWordArray(line);
	}
	
	public void doInference(HMMBase model) {
		//TODO: to save memory free this right after its use
		decodedStates = null;
		this.model = model;
		posteriors = new double[model.nrLayers][][];		
		forwardBackwardList = new ArrayList<ForwardBackward>();
		if (model.hmmType == HMMType.LOG_SCALE) {
			logLikelihood = 0;
			for(int l=0; l<model.nrLayers; l++) {
				ForwardBackwardLog tempFB = new ForwardBackwardLog(model, this, l);
				//find initial posteriors
				tempFB.doInference();
				forwardBackwardList.add(tempFB);
				logLikelihood += tempFB.logLikelihood;
			}
		} else {
			throw new UnsupportedOperationException("Not implemented");
		}		
	}

	public void clearInference() {
		forwardBackwardList.clear();
		forwardBackwardList = null;
		observationCache = null;
	}
	
	/*
	 * TODO: decode viterbi
	 */
	public void decode() {
		decodeViterbi();
	}
	
	/*
	 * decodes viterbi states in each layer
	 */
	public void decodeViterbi() {
		decodedStates = new int[model.nrLayers][];
		for(int l=0; l<model.nrLayers; l++) {
			int[] decoded = new int[T];
			double[][] probLattice = new double[T][model.nrStates];
			int[][] stateLattice = new int[T][model.nrStates];
			for(int i=0; i<model.nrStates; i++) {
				double init = model.param.initial.get(l).get(i, 0);
				double obs = varParamObs.shi[l][0][i];
				probLattice[0][i] = init + obs;			
			}
			double maxValue = -Double.MAX_VALUE;
			int maxIndex = -1;
			for(int t=1; t<T; t++) {
				for(int j=0; j<model.nrStates; j++) {
					double obs = varParamObs.shi[l][t][j];
					maxValue = -Double.MAX_VALUE;
					maxIndex = -1;
					for(int i=0; i<model.nrStates; i++) {
						double value = probLattice[t-1][i] + model.param.transition.get(l).get(j, i) + obs;
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
			for(int i=0; i<model.nrStates; i++) {
				if(probLattice[T-1][i] > maxValue) {
					maxValue = probLattice[T-1][i];
					decoded[T-1] = i;
				}
			}
			//MyArray.printTable(stateLattice);
			//backtrack
			for(int t=T-2; t>=0; t--) {
				decoded[t] = stateLattice[t+1][decoded[t+1]];			
			}
			decodedStates[l] = decoded;
		}
	}
	
	/*
	 * returns log(ObservationProb) or ObservationProb depending on the model
	 */
	public double getObservationProbabilityUsingLLModel(int position) {
		if (observationCache == null) {
			observationCache = new double[T];
			for (int t = 0; t < T; t++) {
				double[] conditionalVector = getConditionalVector(t);
				int observationIndex = this.words[t][0];
				double[] expWeightObservation = model.param.expWeightsCache[observationIndex];
				double numerator = MathUtils.expDot(expWeightObservation, conditionalVector);
				double result = numerator / getExactNormalizer(t, model.param.expWeightsCache);
				if(model.hmmType == HMMType.LOG_SCALE) {
					result = Math.log(result);
				}
				observationCache[t] = result;								
			}
		}
		return observationCache[position];
	}
	
	public String getConditionalString(int t) {
		StringBuffer sb = new StringBuffer();
		for(int l=0; l<model.nrLayers; l++) {
			sb.append(decodedStates[l][t]);
			if(l != model.nrLayers - 1) {
				sb.append("+");
			}
		}
		return sb.toString();
	}
	
	
	/*
	 * gets conditional vector using viterbi decoded (state is fixed for a time t)
	 */
	public double[] getConditionalVector(int t){
		double[] conditionalVector = new double[model.param.weights.conditionalSize];
		//fill the conditionVector
		//conditionalVector[0] = 1.0; //offset
		int index = 0;
		for(int l=0; l<model.nrLayers; l++) {
			for(int i=0; i<nrStates; i++) {
				if(i == decodedStates[l][t]) {
					conditionalVector[index] = 1.0;
				} else {
					conditionalVector[index] = 0.0;
				}
				index++;
			}
		}
		return conditionalVector;
	}
	
	public double getConditionalLogLikelihoodUsingViterbi(double[][] expWeights) {
		double cll = 0.0;
		for(int t=0; t<T; t++) {
			double[] conditionalVector = getConditionalVector(t);
			int observationIndex = this.words[t][0];
			double numerator = MathUtils.expDot(expWeights[observationIndex], conditionalVector);
			double normalizer = getExactNormalizer(t, expWeights);
			cll += Math.log(numerator) - Math.log(normalizer);
		}
		return cll;
	}
	
	public double getExactNormalizer(int position, double[][] expWeights) {
		double Z = 0;
		double[] conditionalVector = getConditionalVector(position);
		String conditionalString = getConditionalString(position);
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
