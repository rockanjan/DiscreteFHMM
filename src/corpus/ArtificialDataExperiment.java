package corpus;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import util.MathUtils;

import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.HMMPowModel;
import model.train.EM;
import config.Config;

/*
 * generates artificial data based on some fixed model
 */
public class ArtificialDataExperiment {
	static HMMBase model;
	static Corpus corpus;
	static int artificialVocabSize = 4;
	static int sampleSize = 20;
	static int T = 20; //sample length
	
	static int repeatCount = 15; //15 sets of experiments
	public static void main(String[] args) throws FileNotFoundException {
		Config.nrLayers = 5;
		Config.numStates = 3;
		
		//start
		double[] trainActualLL = new double[repeatCount];
		double[] trainTrainedLL = new double[repeatCount];
		double[] testActualLL = new double[repeatCount];
		double[] testTrainedLL = new double[repeatCount];
		
		for(int r=0; r<repeatCount; r++) {
			Config.random = new Random();
			Config.vocabThreshold = 0;
			corpus = new Corpus("\\s+", Config.vocabThreshold);
			corpus.createArtificialVocab(artificialVocabSize);
			//corpus.corpusVocab.get(0).debug();
			corpus.corpusVocab.get(0).writeDictionary("artificial_vocab.txt");
			model = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
			corpus.model = model;
			model.initializeRandom(Config.random);
			model.initializeZerosToBest();
			model.param.expWeights = model.param.weights.getCloneExp();
			generateArtificialTest();
			generateArtificialTrain();
			double[] actual = getLLActual();
			double[] trained = getLLTrained();
			trainActualLL[r] = actual[0];
			trainTrainedLL[r] = trained[0];
			testActualLL[r] = actual[1];
			testTrainedLL[r] = trained[1];
		}
		System.out.println("TrainActual: mean = " + MathUtils.getMean(trainActualLL) + " std = " + MathUtils.getStd(trainActualLL));
		System.out.println("TestActual: mean = " + MathUtils.getMean(testActualLL) + " std = " + MathUtils.getStd(testActualLL));
		
		System.out.println("TrainTrained: mean = " + MathUtils.getMean(trainTrainedLL) + " std = " + MathUtils.getStd(trainTrainedLL));
		System.out.println("TestTrained: mean = " + MathUtils.getMean(testTrainedLL) + " std = " + MathUtils.getStd(testTrainedLL));
		
		
	}
	
	public static double[] getLLActual() throws FileNotFoundException {
		double[] result = new double[2];
		//get actual model LL
		HMMPowModel powModelActual = new HMMPowModel(model);
		double varLLTrainActual = 0.0;
		for(int n=0; n<sampleSize; n++) {
			varLLTrainActual += Corpus.trainInstanceList.get(n).getVariationalLL(powModelActual);
		}
		//actual test
		double varLLTestActual = 0.0;
		for(int n=0; n<sampleSize; n++) {
			varLLTestActual += Corpus.testInstanceList.get(n).getVariationalLL(powModelActual);
		}
		result[0] = varLLTrainActual;
		result[1] = varLLTestActual;
		return result;
	}
	
	public static double[] getLLTrained() throws FileNotFoundException {
		/********** Begin Trained ************/
		//get actual model LL
		double[] result = new double[2];
		Config.sampleSizeEStep = -1; //use all samples in training
		Config.numIter = 100;
		Config.random = new Random();
		HMMBase trainModel = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		corpus.model = trainModel;
		trainModel.initializeRandom(Config.random);
		trainModel.initializeZerosToBest();
		EM em = new EM(Config.numIter, corpus, model);
		em.start();
		HMMPowModel powModelTrained = new HMMPowModel(trainModel);
		//trained train
		double varLLTrainTrained = 0.0;
		for(int n=0; n<sampleSize; n++) {
			varLLTrainTrained += Corpus.trainInstanceList.get(n).getVariationalLL(powModelTrained);
		}
		//trained test
		double varLLTestTrained = 0.0;
		for(int n=0; n<sampleSize; n++) {
			varLLTestTrained += Corpus.testInstanceList.get(n).getVariationalLL(powModelTrained);
		}
		result[0] = varLLTrainTrained;
		result[1] = varLLTestTrained;
		return result;
	}
	
	public static double[] getObservationDistribution(int[] states) {
		double obs[] = new double[artificialVocabSize];
		Arrays.fill(obs, 1.0);
		double normalizer = 0.0;
		for(int v=0; v<artificialVocabSize; v++) {
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
		for(int v=0; v<artificialVocabSize; v++) {
			obs[v] /= normalizer;
		}
		return obs;
	}
	
	public static void generateArtificialTrain() throws FileNotFoundException {
		// generate random training data based on the model
		// based on urn and ball model (Rabiner's HMM Tutorial)
		Corpus.trainInstanceList = new InstanceList();
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
	
	public static void generateArtificialTest() throws FileNotFoundException {
		// generate random test data based on the model
		// based on urn and ball model (Rabiner's HMM Tutorial)
		Corpus.testInstanceList = new InstanceList();
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
			Corpus.testInstanceList.add(instance);
			Corpus.testInstanceList.numberOfTokens += instance.words.length;
		}
		PrintWriter testCorpusWriter = new PrintWriter("artificial_test.txt");
		for(int n=0; n<sampleSize; n++) {
			Instance instance = Corpus.testInstanceList.get(n);
			for(int t=0; t<instance.T; t++) {
				testCorpusWriter.print(instance.words[t][0]);
				if(t != instance.T - 1) {
					testCorpusWriter.print(" ");
				}
			}
			testCorpusWriter.println();
		}
		testCorpusWriter.close();
	}
}
