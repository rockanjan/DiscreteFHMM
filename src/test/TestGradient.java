package test;

import java.io.IOException;
import java.util.Random;

import corpus.Corpus;
import corpus.Instance;
import util.MyArray;
import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.HMMType;
import model.param.HMMParamNoFinalStateLog;
import model.train.EM;
import model.train.LogLinearWeightsOptimizable;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;

public class TestGradient {
	/** user parameters **/
	static String delimiter = "\\+";
	static int numIter;
	static long seed = 37;
	
	static String trainFile;
	static String vocabFile;
	static String testFile;
	static String outFolderPrefix;
	static int numStates; 	
	static int vocabThreshold = 5; //only above this included
	static HMMBase model;
	static Corpus corpus;
	
	static int oneTimeStepObsSize; //number of elements in observation e.g. word|hmm1|hmm2  has 3
	
	public static void main(String[] args) throws IOException {
		int recursionSize = 1;
		outFolderPrefix = "out/";
		numStates = 10;
		numIter = 20;
		//String trainFile = "/home/anjan/workspace/HMM/data/test.txt.SPL";
		String trainFile = "/home/anjan/workspace/HMM/data/simple_corpus_sorted.txt";
		String testFile = trainFile;
		HMMType modelType = HMMType.LOG_SCALE;
		vocabFile = trainFile;
		corpus = new Corpus("\\s+", vocabThreshold);
		Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(vocabFile);
		//TRAIN
		corpus.readVocab(vocabFile);
		corpus.readTrain(trainFile);
		corpus.readTest(testFile);
		model = new HMMNoFinalStateLog(numStates, corpus);
		//Random random = new Random(seed);
		Random random = new Random();
		model.initializeRandom(random);
		model.computePreviousTransitions();
		model.initializeZerosToBest();
		
		double[] initParams = MyArray.createVector(model.param.weights.weights);
		LogLinearWeightsOptimizable optimizable = new LogLinearWeightsOptimizable(initParams, corpus);
		
		HMMParamNoFinalStateLog expectedCounts = new HMMParamNoFinalStateLog(model);
		expectedCounts.initializeZeros();
		for (int n = 0; n < corpus.trainInstanceList.size(); n++) {
			Instance instance = corpus.trainInstanceList.get(n);
			instance.doInference(model);
			instance.forwardBackward.addToCounts(expectedCounts);
			instance.clearInference();
		}
		optimizable.checkGradientComputation();
	}
	
}
