package corpus;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.MathUtils;
import config.Config;
import config.LastIter;
import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.HMMType;
import model.param.LogLinearWeights;

/*
 * generates artificial data based on some fixed model
 */
public class ArtificialDataExperiment {
	static HMMBase model;
	static Corpus corpus;
	static int artificial_vocab_size = 4;
	static int sampleSize = 20;
	static int T = 20; //sample length
	static List<StateCombination> stateCombinationList;
	static int stateCombinationSize;
	public static void main(String[] args) throws FileNotFoundException {
		Config.nrLayers = 2;
		Config.numStates = 5;
		stateCombinationSize = (int) Math.pow(Config.numStates, Config.nrLayers);
		populateStateCombination();
		corpus = new Corpus("\\s+", Config.vocabThreshold);
		corpus.createArtificialVocab(artificial_vocab_size);
		//corpus.corpusVocab.get(0).debug();
		corpus.corpusVocab.get(0).writeDictionary("vocab.txt");
		Corpus.trainInstanceList = new InstanceList();
		model = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		corpus.model = model;
		model.initializeRandom(Config.random);
		model.initializeZerosToBest();
		model.param.expWeights = model.param.weights.getCloneExp();
		// generate random training data based on the model
		// based on urn and ball model (Rabiner's HMM Tutorial)
		DiscreteSampler sampler;
		for(int n=0; n<sampleSize; n++) {
			int[] words = new int[T];
			//initial
			int[] stateVector = new int[Config.nrLayers];
			for(int m=0; m<Config.nrLayers; m++) {
				sampler = new DiscreteSampler(model.param.initial.get(m).getDistributionGivenState(0));
				stateVector[m] = sampler.sample();
			}
			//distribution for y at t=0 for the given sampled initial states
			sampler = new DiscreteSampler(getObservationDistribution(stateVector));
			words[0] = sampler.sample();
			//transition
			for(int t=1; t<T; t++) {
				int[] prevStateVector = stateVector.clone();
				//transition
				for(int m=0; m<Config.nrLayers; m++) {
					sampler = new DiscreteSampler(model.param.transition.get(m).getDistributionGivenState(prevStateVector[m]));
					stateVector[m] = sampler.sample();
				}
				//observation
				sampler = new DiscreteSampler(getObservationDistribution(stateVector));
				words[t] = sampler.sample();
			}
			Instance instance = new Instance(corpus, words);
			Corpus.trainInstanceList.add(instance);
			Corpus.trainInstanceList.numberOfTokens += instance.words.length;
		}
		PrintWriter trainCorpusWriter = new PrintWriter("artificial_train.txt");
		for(int n=0; n<sampleSize; n++) {
			Instance instance = Corpus.trainInstanceList.get(n);
			for(int t=0; t<instance.T; t++) {
				trainCorpusWriter.print(instance.words[t][0]);
				if(t != instance.T - 1) {
					trainCorpusWriter.print(" ");
				}
			}
			trainCorpusWriter.println();
		}
		trainCorpusWriter.close();
	}
	
	public static double[] getObservationDistribution(int[] states) {
		double obs[] = new double[artificial_vocab_size];
		Arrays.fill(obs, 1.0);
		double normalizer = 0.0;
		for(int v=0; v<artificial_vocab_size; v++) {
			for(int m=0; m<Config.nrLayers; m++) {
				for(int k=0; k<Config.numStates; k++) {
					if(states[m] == k) {
						obs[v] *= model.param.expWeights.get(m, k, v);
					}
				}
			}
			normalizer += obs[v];
		}
		//normalize
		for(int v=0; v<artificial_vocab_size; v++) {
			obs[v] /= normalizer;
		}
		return obs;
	}
	
	public static void populateStateCombination() {
		stateCombinationList = new ArrayList<StateCombination>();
		for(int s=0; s<stateCombinationSize; s++) {
			stateCombinationList.add(getStateCombination(s));
		}
		//debugStateCombination();
	}
	
	public static StateCombination getStateCombination(int index) {
		int[] stateCombinationVector = new int[Config.nrLayers];
		int stateIndex = 0;
		for(int pow=Config.nrLayers-1; pow>=0; pow--) {
			int divisor = (int) Math.pow(Config.numStates, pow);
			int state = index / divisor; //quotient
			stateCombinationVector[stateIndex++] = state;
			index = index - divisor * state; //remainder 
		}
		StateCombination stateCombination = new StateCombination(stateCombinationVector);
		return stateCombination;
	}
	
	public static void debugStateCombination() {
		for(int i=0; i<stateCombinationList.size(); i++) {
			System.out.println(i + " --> " + stateCombinationList.get(i));
		}
	}
}
