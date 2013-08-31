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
	public static int sampleSizeEStep = 250;
	public static int sampleSizeMStep = 250;
	public final static int numIter = 10;
	public final static int nrLayers = 20;
	public final static int numStates = 2;
	public final static int USE_THREAD_COUNT = 4;
	public final static int vocabThreshold = 1;
	
	public static final String baseDirData = "data/";
	public static final String baseDirModel = "out/model/";
	public static final String baseDirDecode = "out/decoded/";
	
	public static String trainFile = "combined.txt.SPL";
	public static String vocabFile = trainFile;
	public static String testFile;
	public static String devFile;	
	
	public static String outFileTrain = trainFile + ".decoded";
	public static String outFileTest = testFile + ".decoded";
	public static String outFileDev = devFile + ".decoded";
	
	public static int FEATURE_PARTITION_CACHE_SIZE = 100000;
	public static int maxFrequentConditionals = 100000;
	
	//EM related
	// convergence criteria
	public static double precision = 1e-4;
	public static int maxConsecutiveDecreaseLimit = 50;
	public static int maxConsecutiveConvergeLimit = 3;
	public static int mStepIter = 20; 
	//LBFGS
	public static double c2 = 0.01; // L2-regularizer constant (higher means higher penalty)
	
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
		sb.append("\n---------params------------");
		System.out.println(sb.toString());
	}
}
