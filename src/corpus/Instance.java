package corpus;

import java.util.ArrayList;
import java.util.List;

import config.Config;

import model.HMMBase;
import model.HMMType;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;
import model.inference.VariationalParam;
import model.param.LogLinearWeights;
import util.MathUtils;
import util.SmoothWord;

public class Instance {
	
	public VariationalParam varParam;
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public List<ForwardBackward> forwardBackwardList;

	// TODO: might change if we consider finalState hmm
	public static int nrStates = Config.numStates;
	public int unknownCount;
	public HMMBase model;

	public static int FEATURE_PARTITION_CACHE_SIZE = 10000;

	// m,t,k
	public double[][][] posteriors;

	// m, t
	public int[][] decodedStates;

	public double[] observationCache;

	public double logLikelihood;
	public double stateObjective;
	public double observationObjective;
	public double jointObjective; //expected joint log-likelihood E[P(S,Y)]
	
	public double posteriorDifference = 0;
	public double posteriorDifferenceMax = 0;

	public Instance(Corpus c, String line) {
		this.c = c;
		unknownCount = 0;
		// read from line
		populateWordArray(line);
	}

	public void doInference(HMMBase model) {
		// TODO: to save memory free this right after its use
		decodedStates = null;
		this.model = model;
		if(posteriors == null) {
			posteriors = new double[model.nrLayers][][];
		}
		logLikelihood = 0;
		stateObjective = 0;
		observationObjective = 0;
		jointObjective = 0;
		if(forwardBackwardList == null) {
			forwardBackwardList = new ArrayList<ForwardBackward>();
			for (int l = 0; l < model.nrLayers; l++) {
				ForwardBackwardLog tempFB = new ForwardBackwardLog(model, this, l);
				tempFB.doInference();
				forwardBackwardList.add(tempFB);
				logLikelihood += tempFB.logLikelihood;
			}
		}
		else {
			for (int l = 0; l < model.nrLayers; l++) {
				ForwardBackwardLog tempFB = (ForwardBackwardLog) forwardBackwardList.get(l);
				tempFB.doInference();
				logLikelihood += tempFB.logLikelihood;
			}
		}
		//compute the final objective
		//state objective are already calculated
		//exp weights are already cached right inside updateCounts()
		observationObjective = getConditionalLogLikelihoodSoft(model.param.weights.weights, model.param.expWeights.weights);
		jointObjective = stateObjective + observationObjective;
	}

	public void clearInference() {
		forwardBackwardList.clear();
		forwardBackwardList = null;
		observationCache = null;
		posteriors = null;
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
		decodedStates = new int[c.model.nrLayers][];
		for (int l = 0; l < c.model.nrLayers; l++) {
			int[] decoded = new int[T];
			double[][] probLattice = new double[T][c.model.nrStates];
			int[][] stateLattice = new int[T][c.model.nrStates];
			for (int i = 0; i < c.model.nrStates; i++) {
				double init = c.model.param.initial.get(l).get(i, 0);
				double obs = varParam.varParamObs.shi[l][0][i];
				probLattice[0][i] = init + obs;
			}
			double maxValue = -Double.MAX_VALUE;
			int maxIndex = -1;
			for (int t = 1; t < T; t++) {
				for (int j = 0; j < c.model.nrStates; j++) {
					double obs = varParam.varParamObs.shi[l][t][j];
					maxValue = -Double.MAX_VALUE;
					maxIndex = -1;
					for (int i = 0; i < c.model.nrStates; i++) {
						double value = probLattice[t - 1][i]
								+ c.model.param.transition.get(l).get(j, i) + obs;
						if (value > maxValue) {
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
			for (int i = 0; i < c.model.nrStates; i++) {
				if (probLattice[T - 1][i] > maxValue) {
					maxValue = probLattice[T - 1][i];
					decoded[T - 1] = i;
				}
			}
			// MyArray.printTable(stateLattice);
			// backtrack
			for (int t = T - 2; t >= 0; t--) {
				decoded[t] = stateLattice[t + 1][decoded[t + 1]];
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
				double[] expWeightObservation = c.model.param.expWeights.weights[observationIndex];
				double numerator = MathUtils.expDot(expWeightObservation,
						conditionalVector);
				double result = numerator
						/ getExactNormalizer(t, c.model.param.expWeights.weights);
				if (c.model.hmmType == HMMType.LOG_SCALE) {
					result = Math.log(result);
				}
				observationCache[t] = result;
			}
		}
		return observationCache[position];
	}

	public String getConditionalString(int t) {
		StringBuffer sb = new StringBuffer();
		for (int l = 0; l < Config.nrLayers; l++) {
			sb.append(decodedStates[l][t]);
			if (l != Config.nrLayers - 1) {
				sb.append("+");
			}
		}
		return sb.toString();
	}

	/*
	 * gets conditional vector using viterbi decoded (state is fixed for a time
	 * t)
	 */
	public double[] getConditionalVector(int t) {
		double[] conditionalVector = new double[c.model.param.weights.conditionalSize];
		// fill the conditionVector
		// conditionalVector[0] = 1.0; //offset
		int index = 0;
		for (int l = 0; l < c.model.nrLayers; l++) {
			for (int i = 0; i < nrStates; i++) {
				if (i == decodedStates[l][t]) {
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
		for (int t = 0; t < T; t++) {
			double[] conditionalVector = getConditionalVector(t);
			int observationIndex = this.words[t][0];
			double numerator = MathUtils.expDot(expWeights[observationIndex],
					conditionalVector);
			double normalizer = getExactNormalizer(t, expWeights);
			cll += Math.log(numerator) - Math.log(normalizer);
		}
		return cll;
	}
	
	public double getConditionalLogLikelihoodSoft(double[][] weights, double[][] expWeights) {
		double cll = 0.0;
		for (int t = 0; t < T; t++) {
			for(int m=0; m < model.nrLayers; m++) {
				for(int k=0; k<model.nrStates; k++) {
					cll += posteriors[m][t][k] * weights[words[t][0]][LogLinearWeights.getIndex(m, k)];
				}
			}
			double vocabSize= weights.length;
			for(int y=0; y<vocabSize; y++) {
				double prod = 1.0;
				for(int m=0; m<model.nrLayers; m++) {
					double dot = 0;
					for(int k=0; k<model.nrStates; k++) {
						dot += posteriors[m][t][k] * expWeights[y][LogLinearWeights.getIndex(m, k)];
					}
					prod *= dot;
					MathUtils.check(prod);
					if(prod == 0) {
						throw new RuntimeException("underflow");
					}
				}
				cll -= prod;
			}
			cll = cll + 1; // logx <= x - 1			
		}
		return cll;
	}
	
	

	public double getExactNormalizer(int position, double[][] expWeights) {
		double Z = 0;
		double[] conditionalVector = getConditionalVector(position);
		String conditionalString = getConditionalString(position);
		if (InstanceList.featurePartitionCache.containsKey(conditionalString)) {
			return InstanceList.featurePartitionCache.get(conditionalString);
		}
		for (int i = 0; i < Corpus.corpusVocab.get(0).vocabSize; i++) {
			double numerator = MathUtils.expDot(expWeights[i],
					conditionalVector);
			Z += numerator;
		}
		if (InstanceList.featurePartitionCache.size() < FEATURE_PARTITION_CACHE_SIZE) {
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
			if (obsElements.length != Corpus.oneTimeStepObsSize) {
				throw new RuntimeException(
						"One timestep observation size from vocab : "
								+ Corpus.oneTimeStepObsSize
								+ " from instance: " + obsElements.length);
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
			if (wordId == 0) {
				unknownCount++;
			}
			words[i][0] = wordId;
			// for hmm states as observations
			for (int j = 1; j < obsElements.length; j++) {
				String obsElement = obsElements[j];
				int obsElementId = Corpus.corpusVocab.get(j).getIndex(
						obsElement);
				words[i][j] = obsElementId;
			}
		}
	}
}
