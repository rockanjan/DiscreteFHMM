package program;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import util.MyArray;
import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.inference.Decoder;
import model.param.HMMParamBase;
import model.param.HMMParamNoFinalStateLog;
import model.train.EM;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class Main {
	
	/** user parameters **/
	static String delimiter = "\\+";
	static int numIter;
	static long seed = 37;
	
	static String trainFile;
	static String vocabFile;
	static String testFile;
	static String outFolderPrefix;
	static int numStates; 	
	static int vocabThreshold = 1; //only above this included
	static HMMBase model;
	static Corpus corpus;
	public static int currentRecursion;
	public static int sampleSizeEStep;
	public static int sampleSizeMStep;
	
	static int oneTimeStepObsSize; //number of elements in observation e.g. word|hmm1|hmm2  has 3
	
	static int recursionSize = 10;
	/** user parameters end **/
	public static void main(String[] args) throws IOException {
		numStates = 2;
		recursionSize = 100;
		train();
		//String testFilenameBase = "/home/anjan/workspace/HMM/out/decoded/simple_corpus_sorted.txt";
		//decodeFromPlainText(testFilenameBase, recursionSize);
	}
	
	public static void train() throws IOException {
		outFolderPrefix = "out/";
		numStates = 2;
		numIter = 30;
		String trainFileBase;
		String testFileBase;
		trainFileBase = "out/decoded/train.txt.SPL";
		testFileBase = "out/decoded/test.txt.SPL";
		
//		trainFileBase = "out/decoded/simple_corpus_sorted.txt";
//		testFileBase = "out/decoded/simple_corpus_sorted.txt";
		
		double[][] previousRecursionWeights = null;
		
	
		for(int currentRecursion=0; currentRecursion<recursionSize; currentRecursion++) {
			sampleSizeEStep = 5000;
			sampleSizeMStep = 100;
			System.out.println("RECURSION: " + currentRecursion);
			System.out.println("-----------------");
			trainFile = trainFileBase + "." + currentRecursion;
			testFile = testFileBase + "." + currentRecursion;
			vocabFile = trainFile;
			String outFileTrain = trainFileBase + "." + (currentRecursion+1);
			String outFile = testFileBase + "." + (currentRecursion+1);
			printParams();
			corpus = new Corpus("\\s+", vocabThreshold);
			Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(vocabFile);
			//TRAIN
			corpus.readVocab(vocabFile);
			corpus.readTrain(trainFile);
			corpus.readTest(testFile);
			model = new HMMNoFinalStateLog(numStates, corpus);
			Random random = new Random(seed);
			model.initializeRandom(random);
			model.computePreviousTransitions();
			model.initializeZerosToBest();
			//initialize weights with previous recursion weights
			if(previousRecursionWeights != null) {
				model.param.initializeWeightsFromPreviousRecursion(previousRecursionWeights);
			}
			EM em = new EM(numIter, corpus, model);
			em.start();
			model.saveModel(currentRecursion);
			//store weights to assign for the next recursion
			previousRecursionWeights = MyArray.getCloneOfMatrix(model.param.weights.weights);
			test(model, corpus.testInstanceList, outFile);		
			test(model, corpus.trainInstanceList, outFileTrain);
		}	
	}
	
	public static void decodeFromPlainText(String testFileBase, int numberOfRecursions) throws IOException {
		for(currentRecursion = 0; currentRecursion < numberOfRecursions; currentRecursion++) {
			corpus = new Corpus("\\s+", vocabThreshold);
			model = new HMMNoFinalStateLog(numStates, corpus);
			model.loadModel(currentRecursion); //also reads the vocab dictionaries
			String testFile = testFileBase + "." + currentRecursion;
			corpus.readTest(testFile);
			String outFile = testFileBase + "." + (currentRecursion+1);
			test(model, corpus.testInstanceList, outFile);
		}
	}
	
	public static void test(HMMBase model, InstanceList instanceList, String outFile) {
		System.out.println("Decoding Data");
		Decoder decoder = new Decoder(model);
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(outFile));
			PrintWriter pwSimple = new PrintWriter(new FileWriter(outFile + ".simple")); //word and latest hmmstate (no intermediate hmm states)
			PrintWriter pwSimpleAll = new PrintWriter(new FileWriter(outFile + ".simple.all")); //word and intermediate hmm states and final
			for(int n=0; n<instanceList.size(); n++) {
				Instance instance = instanceList.get(n);
				int[] decoded = decoder.viterbi(instance);
				for(int t=0; t<decoded.length; t++) {
					String word = instance.getWord(t);
					int state = decoded[t];
					pwSimple.println(word + " " + state);
					pwSimpleAll.print(word + "\t");
					for(int k=0; k<corpus.oneTimeStepObsSize; k++) {
						pw.print(corpus.corpusVocab.get(k).indexToWord.get(instance.words[t][k]));
						pw.print("|");
						if(k != 0) {
							//except the word itself
							pwSimpleAll.print(corpus.corpusVocab.get(k).indexToWord.get(instance.words[t][k]));
							pwSimpleAll.print("|");
						}
					}
					pw.print(state);
					if(t != decoded.length-1) {
						pw.print(" ");
					}
					pwSimpleAll.print(state);
					pwSimpleAll.println();
				}
				pwSimpleAll.println();
				pwSimple.println();
				pw.println();
			}
			pwSimple.println();
			pw.close();
			pwSimple.close();
			pwSimpleAll.close();
		} catch (IOException e) {
			System.err.format("Could not open file for writing %s\n", outFile);
			e.printStackTrace();
		}
		System.out.println("Finished decoding");
	}
	
	public static void testPosteriorDistribution(HMMBase model, InstanceList instanceList, String outFile) {
		System.out.println("Decoding Posterior distribution");
		Decoder decoder = new Decoder(model);
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(outFile));
			for(int n=0; n<instanceList.size(); n++) {
				Instance instance = instanceList.get(n);
				double[][] decoded = decoder.posteriorDistribution(instance);
				for(int t=0; t<decoded.length; t++) {
					String word = instance.getWord(t);
					pw.print(word + " ");
					for(int i=0; i<decoded[t].length; i++) {
						pw.print(decoded[t][i]);
						if(i != model.nrStates) {
							pw.print(" ");
						}
					}
					pw.println();
				}
				pw.println();
			}
			pw.close();
		} catch (IOException e) {
			System.err.format("Could not open file for writing %s\n", outFile);
			e.printStackTrace();
		}
		System.out.println("Finished decoding");
	}
	
	
	public static void printParams() {
		StringBuffer sb = new StringBuffer();
		sb.append("Train file : " + trainFile);
		sb.append("\nVocab file : " + vocabFile);
		sb.append("\nTest file : " + testFile);
		sb.append("\noutFolderPrefix : " + outFolderPrefix);
		sb.append("\nIterations : " + numIter);
		sb.append("\nNumStates : " + numStates);
		System.out.println(sb.toString());
	}	
}
