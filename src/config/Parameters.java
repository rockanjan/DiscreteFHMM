package config;

import java.util.Random;

/*
 * hyperparams
 */
public class Parameters {
	public final static long seed = 4321;
	public static int sampleSizeEStep = 5000;
	public static int sampleSizeMStep = 5000;
	public final static int numIter = 50;
	public final static int nrLayers = 5;
	public final static int numStates = 2;
	public final static int USE_THREAD_COUNT = 6;
	public final static int vocabThreshold = 1;
	public final static String delimiter = "\\+";
	public final static String obsDelimiter = "\\|"; //separator of multiple observation elements at one timestep
	
	public static String trainFile;
	public static String vocabFile;
	public static String testFile;
	public static String devFile;	
	public static String outFolderPrefix;
	public static int VOCAB_UPDATE_COUNT = 0;
	public static int FEATURE_PARTITION_CACHE_SIZE = 10000;
	public static int maxTokensToProcessForFrequentConditionals = 1000000;
	public static int maxFrequentConditionals = 100000;
	
	
	public static Random random = new Random(seed);
}
