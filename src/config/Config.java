package config;

import java.util.Random;

import corpus.Corpus;
import corpus.InstanceList;

/*
 * hyperparams
 */
public class Config {
	public final static long seed = 4321;
	public static Random random = new Random(seed);
	public final static int numIter = 100;
	public final static int nrLayers = 7;
	public final static int numStates = 20;
	public final static int USE_THREAD_COUNT = 8;
	public final static int vocabThreshold = 2;
	
	public static final String baseDirData = "data/";
	public static final String baseDirModel = "out/model/";
	public static final String baseDirDecode = "out/decoded/";
	
	public static String trainFile = "pos_ul.10k.notag";
	public static String vocabFile = "pos_ul.10k.notag";
	public static String testFile = "pos_ul.test.notag";
	public static String devFile = "pos.dev.txt";	
	
	public static String outFileTrain = trainFile + ".decoded";
	public static String outFileTest = testFile + ".decoded";
	public static String outFileDev = devFile + ".decoded";
	
	public static int FEATURE_PARTITION_CACHE_SIZE = 100000;
	public static int maxFrequentConditionals = 100000;

	public static int variationalIter = 3;
	
	//EM related
	// convergence criteria
	public static double precision = 1e-4;
	public static int maxConsecutiveDecreaseLimit = 5;
	public static int maxConsecutiveConvergeLimit = 3;
	public static int mStepIter = 20;
	public static int convergenceIterInterval = 5; //after how many iters to check the convergence on dev data
	
	//online learning params
	/*
	 * sample sequentially from the randomized sentences
	 * (so that each sentence is covered exactly once in each epoch)
	 * or: sample randomly from the sentences (in expectation, all will be covered) 
	 * (some might be repeated some might be skipped)
	 */
	public static boolean sampleSequential = true;
	public static double alpha = 0.6;
	public static double t0 = 2;
	public static int sampleSizeEStep = 500;
	public static int sampleSizeMStep = sampleSizeEStep;
	public static int sampleDevSize = 100;
	
	//LBFGS
	public static double c2 = 0.01; // L2-regularizer constant (higher means higher penalty)
	
	public static boolean displayDetail = true;
	
	public static void printParams() {
		StringBuffer sb = new StringBuffer();
		sb.append("\n---------params------------");
		sb.append("\nRandom seed : " + seed);
		sb.append("\nTrain file : " + trainFile);
		sb.append("\nVocab file : " + vocabFile);
		sb.append("\nTest file : " + testFile);
		sb.append("\nDev file : " + devFile);
		
		sb.append("\nData Folder : " + baseDirData);
		sb.append("\nModel Folder : " + baseDirModel);
		sb.append("\nDecode Folder : " + baseDirDecode);
		
		sb.append("\nIterations : " + numIter);
		sb.append("\nNumStates : " + numStates);
		sb.append("\nNumLayers : " + nrLayers);
		sb.append("\nThreads : " + USE_THREAD_COUNT);
		sb.append("\nt0 : " + t0);
		sb.append("\nalpha : " + alpha);
		sb.append("\n---------params------------");
		System.out.println(sb.toString());
	}
}
