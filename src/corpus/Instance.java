package corpus;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.management.RuntimeErrorException;

import model.HMMBase;
import model.HMMPowModel;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;
import model.inference.VariationalParam;
import model.param.LogLinearWeights;
import model.param.LogLinearWeightsClass;
import util.MathUtils;
import util.MyArray;
import util.SmoothWord;
import config.Config;

public class Instance {
	public VariationalParam varParam;
	//timestep, layer
	public int[][] tags;
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public List<ForwardBackward> forwardBackwardList;

	// TODO: might change if we consider finalState hmm
	public int unknownCount;
	public HMMBase model;

	// m,t,k
	public double[][][] posteriors;

	// m, t
	public int[][] decodedStates;

	public double[] observationCache;

	public double logLikelihood;
	public double jointObjective;
	
	public double posteriorDifference = 0;
	public double posteriorDifferenceMax = 0;

	public Instance(Corpus c, String line) {
		this.c = c;
		unknownCount = 0;
		// read from line
		populateWordArray(line);
	}
	
	public Instance(Corpus c, int[] words) {
		this.c = c;
		unknownCount = 0;
		T = words.length;
		//System.out.println(Corpus.oneTimeStepObsSize);
		this.words = new int[words.length][Corpus.oneTimeStepObsSize];
		for(int t=0; t<T; t++) {
			this.words[t][0] = words[t];
		}
	}

	public void doInference(HMMBase model) {
		// TODO: to save memory free this right after its use
		decodedStates = null;
		this.model = model;
		if(posteriors == null) {
			posteriors = new double[model.states.length][][];
		}
		logLikelihood = 0;
		
		if(forwardBackwardList == null) {
			forwardBackwardList = new ArrayList<ForwardBackward>();
			for (int l = 0; l < model.states.length; l++) {
				ForwardBackwardLog tempFB = new ForwardBackwardLog(model, this, l);
				tempFB.doInference();
				forwardBackwardList.add(tempFB);
				logLikelihood += tempFB.logLikelihood;
			}
		}
		else {
			for (int l = 0; l < model.states.length; l++) {
				ForwardBackwardLog tempFB = (ForwardBackwardLog) forwardBackwardList.get(l);
				tempFB.doInference();
				logLikelihood += tempFB.logLikelihood;
			}
		}		
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
		decodedStates = new int[c.model.states.length][];
		for (int l = 0; l < c.model.states.length; l++) {
			int[] decoded = new int[T];
			double[][] probLattice = new double[T][c.model.states[l]];
			int[][] stateLattice = new int[T][c.model.states[l]];
			for (int i = 0; i < c.model.states[l]; i++) {
				double init = c.model.param.initial.get(l).get(i, 0);
				double obs = varParam.varParamObs.shi[l][0][i];
				probLattice[0][i] = init + obs;
			}
			double maxValue = -Double.MAX_VALUE;
			int maxIndex = -1;
			for (int t = 1; t < T; t++) {
				for (int j = 0; j < c.model.states[l]; j++) {
					double obs = varParam.varParamObs.shi[l][t][j];
					maxValue = -Double.MAX_VALUE;
					maxIndex = -1;
					for (int i = 0; i < c.model.states[l]; i++) {
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
			for (int i = 0; i < c.model.states[l]; i++) {
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
	/*
	public double getObservationProbabilityUsingLLModel(int position) {
		if (observationCache == null) {
			observationCache = new double[T];
			for (int t = 0; t < T; t++) {
				double[] conditionalVector = getConditionalVector(t);
				int observationIndex = this.words[t][0];
				double[] expWeightObservation = c.model.param.expWeights.weights[observationIndex];
				double numerator = MathUtils.expDot(expWeightObservation,conditionalVector);
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
	*/
	
	public double getJointObjective() {
		//TODO: important: inference should have already been done before calling this
		double jointObjective = 0.0;
		double observationObjectiveWords = getConditionalLogLikelihoodSoftWords(model.param.weights.weights, model.param.expWeights.weights);
		double observationObjectiveClasses = getConditionalLogLikelihoodSoftClasses(model.param.weightsClass.weights,
				model.param.expWeightsClass.weights);
		double stateObjective = 0.0;
		
		//initial
		for(int m=0; m<Config.states.length; m++) {
			for(int k=0; k<Config.states[m]; k++) {
				stateObjective += forwardBackwardList.get(m).getStatePosterior(0, k) * 
						model.param.initial.get(m).get(k, 0);  
			}
		}
		//transition
		for(int m=0; m<Config.states.length; m++) {
			for(int t=0; t<T-1; t++) {
				for(int i=0; i<Config.states[m]; i++) {
					for(int j=0; j<Config.states[m]; j++) {
						double transPosterior = forwardBackwardList.get(m).getTransitionPosterior(i, j, t);
						stateObjective += transPosterior * model.param.transition.get(m).get(j, i); 
					}
				}
			}
		}		
		jointObjective = stateObjective + observationObjectiveClasses + observationObjectiveWords;
		return jointObjective;
		
	}

	
	/*
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
	*/
	
	public double getConditionalLogLikelihoodSoftWords(double[] weights, double[] expWeights) {
		double cll = 0.0;
		for (int t = 0; t < T; t++) {
			for(int m=0; m < model.states.length; m++) {
				for(int k=0; k<model.states[m]; k++) {
					cll += posteriors[m][t][k] * weights[LogLinearWeights.getIndex(m, k, words[t][0])];
				}
			}
			
			int wordCluster = WordClass.wordIndexToClusterIndex.get(this.words[t][0]);
			Set<Integer> wordsInCluster = WordClass.clusterIndexToWordIndices.get(wordCluster);
			double sumOverY = 0;
			for(int y : wordsInCluster) {
				double prod = 1.0;
				for(int m=0; m<model.states.length; m++) {
					double dot = 0;
					for(int k=0; k<model.states[m]; k++) {
						dot += posteriors[m][t][k] * expWeights[LogLinearWeights.getIndex(m, k, y)];
					}
					if(dot == 0) {
						dot = 1e-200;
					}
					prod *= dot;
					//System.out.println("prod = " + prod);
					MathUtils.check(prod);
					if(prod == 0) {
						throw new RuntimeException("underflow");
					}
				}
				sumOverY += prod;
			}
			cll -= Math.log(sumOverY);						
		}
		return cll;
	}
	
	public double getConditionalLogLikelihoodSoftClasses(double[] weightsClass, double[] expWeightsClass) {
		double cll = 0.0;
		for (int t = 0; t < T; t++) {
			int currentCluster = WordClass.wordIndexToClusterIndex.get(this.words[t][0]);
			for(int m=0; m < model.states.length; m++) {
				for(int k=0; k<model.states[m]; k++) {
					cll += posteriors[m][t][k] * weightsClass[LogLinearWeightsClass.getIndex(m, k, currentCluster)];
				}
			}
			
			double sumOverC = 0;
			for(int c=0; c<model.nrClasses; c++) {
				double prod = 1.0;
				for(int m=0; m<model.states.length; m++) {
					double dot = 0;
					for(int k=0; k<model.states[m]; k++) {
						dot += posteriors[m][t][k] * expWeightsClass[LogLinearWeightsClass.getIndex(m, k, c)];
					}
					prod *= dot;
					MathUtils.check(prod);
					if(prod == 0) {
						throw new RuntimeException("underflow");
					}
				}
				sumOverC += prod;
			}
			cll -= Math.log(sumOverC);						
		}
		return cll;
	}
	
	
	/*
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
	*/
	
	/*
	 * Gives variational LL for the instance
	 * takes O(T K^(2*M)) : expensive
	 * should be called after calling variational inference
	 * posterior expectations should not be empty
	 */
	public double getVariationalLL(HMMPowModel powModel) {
		double variationalLL = 0.0;
		/*
		//create alpha table using K^M states
		double[][] alpha = new double[T][powModel.powStateSize];
		//do forward computation
		for(int i=0; i<powModel.powStateSize; i++) {
			double pi = powModel.initial.get(i, 0);
			double obs = powModel.observation.get(words[0][0], i);
			alpha[0][i] = pi + obs; //these prob are in logscale			
		}
		
		//induction
		for(int t = 1; t < T; t++) {
			for(int j=0; j<powModel.powStateSize; j++) {
				double[] expParams = new double[powModel.powStateSize];
				for(int i=0; i<powModel.powStateSize; i++) {
					expParams[i] = alpha[t-1][i] + powModel.transition.get(j, i); 
				}
				double obs;
				obs = powModel.observation.get(words[t][0], j);
				alpha[t][j] = MathUtils.logsumexp(expParams) + obs; 
			}			
		}
		variationalLL = MathUtils.logsumexp(alpha[T-1]);
		if(variationalLL > 0) {
			MyArray.printTable(alpha, "alpha");
			throw new RuntimeException("variationalLL greater than 0");
		}
		*/
		return variationalLL;
	}

	/*
	 * returns the original word at the position
	 */
	public String getWord(int position) {
		return Corpus.corpusVocab.get(0).indexToWord.get(words[position][0]);
	}
	
	public boolean isLabeledLayer(int layer) {
		return (tags != null && layer < Corpus.tagSize);
	}

	public void populateWordArray(String line) {
		String allTimeSteps[] = line.split(Corpus.delimiter);
		T = allTimeSteps.length;
		words = new int[T][Corpus.oneTimeStepObsSize];
		for (int t = 0; t < T; t++) {
			//FORMAT per timestep: word|tag1|...|tagM^obs1|...|obsN : labeled with tags and with extra features
			String oneTimeStep = allTimeSteps[t];
			String[] splittedWordTagAndObsFeatures = oneTimeStep.split(Corpus.obsAndTagSeparator);
			String wordTags = splittedWordTagAndObsFeatures[0];
			String[] wordTagSplitted = wordTags.split(Corpus.obsAndTagDelimiter);
			// original word
			String word = wordTagSplitted[0];
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
			words[t][0] = wordId;
			
			//tags
			if(wordTagSplitted.length > 1) { //has tags
				int numberOfTags = wordTagSplitted.length - 1;
				//when loading test data, tagSize will already have been initialized when loading model
				if(Corpus.tagSize == 0) { //this is the first labeled data found
					Corpus.tagSize = numberOfTags;
					//create tag vocabs
					Corpus.tagVocab = new ArrayList<Vocabulary>();
					for(int d=0; d<numberOfTags; d++) {
						Vocabulary tempVocab = new Vocabulary();
						tempVocab.debug = false;
						tempVocab.smooth = false;
						tempVocab.lower = false;
						tempVocab.vocabThreshold = 0;
						Corpus.tagVocab.add(tempVocab);
					}
				} else { //make sure all labeled data has same tag length
					if(Corpus.tagSize != numberOfTags) {
						throw new RuntimeException("First tag length found = " + Corpus.tagSize + 
								", different found = " + numberOfTags + ", timestep = " + t + ", line = " + line);
					}
				}
				if(t==0) {
					//assign memory for the tags for all timesteps
					tags = new int[T][numberOfTags];
					//by default, assign -1 for all tags 
					//(in case some words have tags and some words do not in a sentence)
					for(int i=0; i<T ;i++) {
						for(int j=0; j<numberOfTags; j++) {
							tags[i][j] = -1;
						}
					}
				}
				//make sure all tokens of a sentence are labeled
				if(tags == null) { //when first token is unlabeled
					throw new RuntimeException("First token not labeled for line : " + line);
				}
				//add to tagDictionary if needed and get the index
				for(int d=0; d<numberOfTags; d++) {
					String tag = wordTagSplitted[d+1]; //offset because first index is word
					int tagIndex = -1;
					if(Corpus.tagVocab.get(d).frozen) {
						tagIndex = Corpus.tagVocab.get(d).getIndex(tag);
					} else {
						tagIndex = Corpus.tagVocab.get(d).addItem(tag);
					}
					tags[t][d] = tagIndex;
				}				
			} else { //if a token is not labeled
				if(tags != null) { //but tag was initialized
					throw new RuntimeException("All tokens must be labeled for line : " + line);
				}
			}
			
			//TODO:
			/*
			//observation features other than word
			if(splittedWordTagAndObsFeatures.length == 2) {
				String obs = splittedWordTagAndObsFeatures[1];
				String[] obsElements = obs.split(Corpus.obsAndTagDelimiter);
				if (obsElements.length != Corpus.oneTimeStepObsSize) {
					throw new RuntimeException(
							"One timestep observation size from vocab : "
									+ Corpus.oneTimeStepObsSize
									+ " from instance: " + obsElements.length);
				}
			}
			for (int j = 1; j < obsElements.length; j++) {
				String obsElement = obsElements[j];
				int obsElementId = Corpus.corpusVocab.get(j).getIndex(
						obsElement);
				words[i][j] = obsElementId;
			}
			*/
		}
	}
}
