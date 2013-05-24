package program;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import util.MyArray;

import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.HMMType;
import model.inference.Decoder;
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
	static int vocabThreshold = 2; //only above this included
	static HMMBase model;
	static Corpus corpus;
	
	static int oneTimeStepObsSize; //number of elements in observation e.g. word|hmm1|hmm2  has 3
	
	/** user parameters end **/
	public static void main(String[] args) throws IOException {
		int recursionSize = 10;
		outFolderPrefix = "out/";
		numStates = 2;
		numIter = 30;
		String trainFileBase = "out/decoded/train.txt.SPL";
		String testFileBase = "out/decoded/test.txt.SPL";
//		String trainFileBase = "out/decoded/simple_corpus_sorted.txt";
//		String testFileBase = "out/decoded/simple_corpus_sorted.txt";
		
		HMMType modelType = HMMType.LOG_SCALE;
		for(int i=0; i<recursionSize; i++) {
			System.out.println("RECURSION: " + i);
			System.out.println("-----------------");
			trainFile = trainFileBase + "." + i;
			testFile = testFileBase + "." + i;
			vocabFile = trainFile;
			String outFileTrain = trainFileBase + "." + (i+1);
			String outFile = testFileBase + "." + (i+1);
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
			//MyArray.printTable(model.param.weights.weights, "Weights");
			model.computePreviousTransitions();
			model.initializeZerosToBest();
			EM em = new EM(numIter, corpus, model);
			//start training with EM
			em.start();
			//MyArray.printTable(model.param.weights.weights, "Final Weights");
			test(model, corpus.testInstanceList, outFile);		
			test(model, corpus.trainInstanceList, outFileTrain);
		}		
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
