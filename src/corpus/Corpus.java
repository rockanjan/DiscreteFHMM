package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import util.SmoothWord;

public class Corpus {
	public static String delimiter = "\\s+";
	public static String obsDelimiter = "\\|"; //separator of multiple observation elements at one timestep
	public static int oneTimeStepObsSize = -1;
	public InstanceList trainInstanceList = new InstanceList();
	// testInstanceList can be empty
	public InstanceList testInstanceList;
	
	public InstanceList trainInstanceEStepSampleList; //sampled sentences for stochastic training
	public InstanceList trainInstanceMStepSampleList; //sampled sentences for stochastic training

	static public ArrayList<Vocabulary> corpusVocab;

	int vocabThreshold;
	
	public int totalWords; 
	
	static Random random = new Random(37);
	
	public List<Double> unigramProbability;
	static DiscreteSampler vocabSampler;
	public static int VOCAB_SAMPLE_SIZE = 10000;

	public Corpus(String delimiter, int vocabThreshold) {
		this.delimiter = delimiter;
		this.vocabThreshold = vocabThreshold;		
	}

	public void readTest(String inFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		int totalUnknown = 0;
		testInstanceList = new InstanceList();
		while ((line = br.readLine()) != null) {
			line = line.trim();
			if (!line.isEmpty()) {
				Instance instance = new Instance(this, line);
				totalUnknown += instance.unknownCount;
				if (instance.words.length != 0) {
					testInstanceList.add(instance);
					totalWords += instance.words.length;
				} else {
					System.out.println("Could not read from test file, line = "
							+ line);
				}
			}
		}
		testInstanceList.numberOfTokens = totalWords;
		System.out.println("Test Instances: " + testInstanceList.size());
		System.out.println("Test token count: " + totalWords);
		System.out.println("Test unknown count : " + totalUnknown);
		br.close();
	}

	public void readTrain(String inFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		totalWords = 0;
		int totalUnknown = 0;
		while ((line = br.readLine()) != null) {
			line = line.trim();
			if (!line.isEmpty()) {
				Instance instance = new Instance(this, line);
				totalUnknown += instance.unknownCount;
				if (instance.words.length != 0) {
					trainInstanceList.add(instance);
					totalWords += instance.words.length;
				} else {
					System.err.println("Could not read from train file, line = "
									+ line);
				}
			}
		}
		trainInstanceList.numberOfTokens = totalWords;
		System.out.println("Train Instances: " + trainInstanceList.size());
		System.out.println("Train token count: " + totalWords);
		System.out.println("Train unknown count : " + totalUnknown);
		br.close();
	}

	public void readVocab(String inFile) throws IOException {
		corpusVocab = new ArrayList<Vocabulary>();
		Vocabulary v = new Vocabulary(); //base vocabulary
		v.vocabReadIndex = 1; //zero reserved for *unk*
		v.vocabThreshold = vocabThreshold;
		corpusVocab.add(v);
		
		//hmm states as vocabs
		for(int i=1; i<oneTimeStepObsSize; i++) {
			Vocabulary vHmmStates = new Vocabulary();
			vHmmStates.vocabThreshold = 0;
			vHmmStates.vocabReadIndex = 0;
			corpusVocab.add(vHmmStates);
		}
		readVocabFromCorpus(inFile);
	}
	
	public void setupSampler() {
		computeUnigramProbabilities();
		vocabSampler = new DiscreteSampler(unigramProbability);
	}
	
	public static int getRandomVocabItem() {
		//return vocabSampler.sample();
		return random.nextInt(corpusVocab.get(0).vocabSize);
	}
	
	public void computeUnigramProbabilities() {
		unigramProbability = new ArrayList<Double>();
		double sum = 0.0;
		int totalFreq = 0;
		for(int i=0; i<corpusVocab.get(0).vocabSize; i++) {
			totalFreq += corpusVocab.get(0).indexToFrequency.get(i);
		}
		for(int i=0; i<corpusVocab.get(0).vocabSize; i++) {
			double prob = 1.0 * corpusVocab.get(0).indexToFrequency.get(i) / totalFreq;
			unigramProbability.add(prob);
			sum += prob;
		}
		if(Math.abs(sum - 1) > 1e-5) {
			throw new RuntimeException("Unigram Probabilities do not sum to 1. sum = " + sum); 
		}
	}
	
	private void readVocabFromCorpus(String filename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		corpusVocab.get(0).wordToIndex.put(Vocabulary.UNKNOWN, 0);
		corpusVocab.get(0).indexToFrequency.put(0, 0);
		corpusVocab.get(0).indexToWord.add(Vocabulary.UNKNOWN);
		
		while( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty()) {
				String allTimeSteps[] = line.split(delimiter);
				for(int i=0; i<allTimeSteps.length; i++) {
					String oneTimeStep = allTimeSteps[i];
					String[] obsElements = oneTimeStep.split(obsDelimiter);
					//original word
					String word = obsElements[0];
					if(corpusVocab.get(0).lower) {
						word = word.toLowerCase();
					}
					if(corpusVocab.get(0).smooth) {
						word = SmoothWord.smooth(word);
					}
					int wordId = corpusVocab.get(0).addItem(word);
					//for hmm states as observations
					for(int j=1; j<obsElements.length; j++) {
						String obsElement = obsElements[j];
						int obsElementId = corpusVocab.get(j).addItem(obsElement);
					}					
				}
			}
		}
		//original word
		corpusVocab.get(0).vocabSize = corpusVocab.get(0).wordToIndex.size();
		System.out.println("Vocab Size before reduction including UNKNOWN : " + corpusVocab.get(0).vocabSize);
		
		if(corpusVocab.get(0).debug) {
			corpusVocab.get(0).debug();
		}
		
		corpusVocab.get(0).reduceVocab(this);
		System.out.println("Vocab Size after reduction including UNKNOWN : " + corpusVocab.get(0).vocabSize);
		
		
		if(corpusVocab.get(0).debug) {
			corpusVocab.get(0).debug();
		}
		
		//hmm states as observations
		for(int i=1; i<oneTimeStepObsSize; i++) {
			corpusVocab.get(i).vocabSize = corpusVocab.get(i).wordToIndex.size();
			System.out.format("Vocab Size for vocab[%d] = %d \n", i, corpusVocab.get(i).vocabSize);
			if(corpusVocab.get(i).debug) {
				System.out.format("Debug for vocab[%d]\n", i);
				corpusVocab.get(i).debug();
			}
		}
		br.close();		
	}

	/*
	public void readVocabFromDictionary(String filename) {
		corpusVocab = new Vocabulary();
		corpusVocab.readVocabFromDictionary(filename);
	}
	*/
	public void saveVocabFile(String filename) {
		for(int i=0; i<oneTimeStepObsSize; i++) {
			PrintWriter dictionaryWriter;
			try {
				dictionaryWriter = new PrintWriter(filename + "." + i);
				int V = corpusVocab.get(i).vocabSize;
				dictionaryWriter.println(V);
				for (int v = 0; v < V; v++) {
					dictionaryWriter.println(corpusVocab.get(i).indexToWord.get(v));
					dictionaryWriter.flush();
				}
				dictionaryWriter.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(-1);
			}
		}
	}
	
	/*
	 * does not make a clone, just the reference
	 */
	public void generateRandomTrainingEStepSample(int size) {
		trainInstanceEStepSampleList = new InstanceList();
		if(trainInstanceList.size() <= size) {
			trainInstanceEStepSampleList.addAll(trainInstanceList);
			trainInstanceEStepSampleList.numberOfTokens = trainInstanceList.numberOfTokens;
		} else {
			ArrayList<Integer> randomInts = new ArrayList<Integer>();			
			for(int i=0; i<trainInstanceList.size(); i++) {
				randomInts.add(i);
			}
			Collections.shuffle(randomInts,random);
			for(int i=0; i<size; i++) {
				Instance instance = trainInstanceList.get(randomInts.get(i));
				trainInstanceEStepSampleList.add(instance);
				trainInstanceEStepSampleList.numberOfTokens += instance.T;
			}			
		}
	}	
	
	/*
	 * should be subset of E-step samples (because the posterior distribution is needed)
	 */
	public void generateRandomTrainingMStepSample(int size) {
		trainInstanceMStepSampleList = new InstanceList();
		if(trainInstanceEStepSampleList.size() <= size) {
			trainInstanceMStepSampleList.addAll(trainInstanceEStepSampleList);
			trainInstanceMStepSampleList.numberOfTokens = trainInstanceEStepSampleList.numberOfTokens;
		} else {
			ArrayList<Integer> randomInts = new ArrayList<Integer>();			
			for(int i=0; i<trainInstanceEStepSampleList.size(); i++) {
				randomInts.add(i);
			}
			Collections.shuffle(randomInts,random);
			for(int i=0; i<size; i++) {
				Instance instance = trainInstanceEStepSampleList.get(randomInts.get(i));
				trainInstanceMStepSampleList.add(instance);
				trainInstanceMStepSampleList.numberOfTokens += instance.T;
			}			
		}
	}	
	
	
	public static int findOneTimeStepObsSize(String filename) {
		int result = -1;
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = "";
			while( (line = br.readLine()) != null) {
				if(line.trim().isEmpty()) continue;
				String[] allObsSplitted = line.split(Corpus.delimiter);
				//System.out.println(allObsSplitted[0]);
				String[] oneTimeStepSplitted = allObsSplitted[0].split(Corpus.obsDelimiter);
				result = oneTimeStepSplitted.length; 
				break;
			}
			br.close();
			if(result < 0) {
				throw new RuntimeException("could not read number of observation elements from vocab file");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("One timestep obs size = " + result);
		return result;
	}	
	
	public static void main(String[] args) throws IOException {
		String inFile = "/home/anjan/workspace/HMM/data/simple_corpus_sorted.txt";
		int vocabThreshold = 1;
		Corpus c = new Corpus("\\s+", vocabThreshold);
		Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(inFile);
		c.readVocab(inFile);
		c.saveVocabFile("/tmp/vocab.txt");
		c.corpusVocab.get(0).readDictionary("/tmp/vocab.txt.0");
		c.readTest(inFile);
		for(int i=0; i<c.testInstanceList.size(); i++) {
			for(int t=0; t<c.testInstanceList.get(i).T; t++) {
				System.out.print(c.testInstanceList.get(i).getWord(t) + " ");
			}
			System.out.println();
		}
	}
}
