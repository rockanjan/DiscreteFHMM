package config;

import java.util.Random;

/*
 * hyperparams
 */
public class Config {
	public static long seed = 1;
	public static Random random = new Random(seed);
	public static int numIter = 500;
	public static int nrLayers = 5;
	public static int numStates = 10;
	public final static int USE_THREAD_COUNT = 8;
	public static int vocabThreshold = 1;
	
	public static final String wordClusterFile = "rcv1.txt.SPL.decoded.cluster";

	public static final String baseDirData = "data/";
	public static final String baseDirModel = "out/model/";
	public static final String baseDirDecode = "out/decoded/";

	public static String trainFile = "rcv1.txt.SPL";
	public static String vocabFile = trainFile;
	public static String testFile = "combined.txt.SPL";
	public static String devFile = "combined.txt.SPL.cleaned";

	public static String outFileTrain = trainFile + ".decoded";
	public static String outFileTest = testFile + ".decoded";
	public static String outFileDev = devFile + ".decoded";

	public static int FEATURE_PARTITION_CACHE_SIZE = 100000;
	public static int maxFrequentConditionals = 100000;

	public static int variationalIter = 10;
	public static double variationalConvergence = 1e-5;

	//EM related
	public static int modelSaveInterval = 10;
	// convergence criteria
	public static double precision = 1e-6;
	public static int maxConsecutiveConvergeLimit = 3;
	public static int mStepIter = 100;
	public static int convergenceIterInterval = 1; //after how many iters to check the convergence on dev data

	//online learning params
	/*
	 * sample sequentially from the randomized sentences
	 * (so that each sentence is covered exactly once in each epoch)
	 * or: sample randomly from the sentences (in expectation, all will be covered)
	 * (some might be repeated some might be skipped)
	 */
	public static boolean sampleSequential = true;
	public static double alpha = 0.5;
	public static double t0 = 2;
	public static int sampleSizeEStep = 10000;
	public static int sampleSizeMStep = sampleSizeEStep;
	public static int sampleDevSize = Integer.MAX_VALUE;
	public static String vocabSamplingType = "unigram";
	public static int VOCAB_SAMPLE_SIZE = 0;

	//LBFGS
	public static double c2 = 0.00001; // L2-regularizer constant (higher means higher penalty)

	public static boolean displayDetail = false;



	public static void printParams() {
		StringBuffer sb = new StringBuffer();
		sb.append("\n---------params------------");
		sb.append("\nRandom seed : " + seed);
		sb.append("\nVocab thres : " + vocabThreshold);
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
