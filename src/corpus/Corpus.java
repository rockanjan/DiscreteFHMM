package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.TreeSet;

import config.Config;

import model.HMMBase;

import program.Main;
import util.SmoothWord;

public class Corpus {
	public static String delimiter = "\\s+";
	public static String obsDelimiter = "\\|"; //separator of multiple observation elements at one timestep
	public static InstanceList trainInstanceList;
	// testInstanceList can be empty
	public static InstanceList testInstanceList;
	public static InstanceList devInstanceList;
	
	//all instances but randomized
	public static InstanceList trainInstanceListRandomized;
	
	public static InstanceList trainInstanceEStepSampleList; //sampled sentences for stochastic training
	public static InstanceList trainInstanceMStepSampleList; //sampled sentences for stochastic training (subset of EStep sample)
	
	public static InstanceList testInstanceSampleList; //sampled sentences for stochastic training
	public static InstanceList devInstanceSampleList; //sampled sentences for stochastic training
	

	static public ArrayList<Vocabulary> corpusVocab;
	
	//public static TreeSet<FrequentConditionalStringVector> frequentConditionals;
	public static ArrayList<FrequentConditionalStringVector> frequentConditionals;

	int vocabThreshold;
	
	public int totalWords; 
		
	public List<Double> unigramProbability;
	static DiscreteSampler vocabSampler;
	public static int VOCAB_SAMPLE_SIZE = 0;
	
	public HMMBase model;
	
	public static int oneTimeStepObsSize = 1; //default is a single word

	public Corpus(String delimiter, int vocabThreshold) {
		this.delimiter = delimiter;
		this.vocabThreshold = vocabThreshold;		
	}
	
	public void readDev(String inFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(inFile));
		String line = null;
		int totalWords = 0;
		int totalUnknown = 0;
		devInstanceList = new InstanceList();
		while ((line = br.readLine()) != null) {
			line = line.trim();
			if (!line.isEmpty()) {
				Instance instance = new Instance(this, line);
				totalUnknown += instance.unknownCount;
				if (instance.words.length != 0) {
					devInstanceList.add(instance);
					totalWords += instance.words.length;
				} else {
					System.out.println("Could not read from dev file, line = " + line);
				}
			}
		}
		devInstanceList.numberOfTokens = totalWords;
		System.out.println("Dev Instances: " + devInstanceList.size());
		System.out.println("Dev token count: " + totalWords);
		System.out.println("Dev unknown count : " + totalUnknown);
		br.close();
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
		trainInstanceList = new InstanceList();
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
	
	public static void clearFrequentConditionals() {
		frequentConditionals = null;
	}
	
	//cache from the training data only
	/*
	 * finds the most frequent conditional vectors 
	 */
	public static void cacheFrequentConditionals() {
		HashMap<FrequentConditionalStringVector, Integer> conditionalCountMap = new HashMap<FrequentConditionalStringVector, Integer>();
		/*
		frequentConditionals = new TreeSet<FrequentConditionalStringVector>(new Comparator<FrequentConditionalStringVector>() {
			@Override
			public int compare(FrequentConditionalStringVector o1,
					FrequentConditionalStringVector o2) {
				return o1.index.compareTo(o2.index);
			}
			
		});
		*/
		frequentConditionals = new ArrayList<FrequentConditionalStringVector>();
		//count frequencies
		for(int n=0; n<Corpus.trainInstanceMStepSampleList.size(); n++) {
			Instance instance = Corpus.trainInstanceMStepSampleList.get(n);
			for(int t=0; t<instance.T; t++) {
				String conditionalString = instance.getConditionalString(t);
				double[] conditionalVector = instance.getConditionalVector(t);
				//MyArray.printVector(conditionalVector, "conditiona");
				FrequentConditionalStringVector fc = new FrequentConditionalStringVector(conditionalString, conditionalVector);
				if(conditionalCountMap.containsKey(fc)) {
					int prevFreq = conditionalCountMap.get(fc);
					conditionalCountMap.put(fc, prevFreq+1);
				} else {
					conditionalCountMap.put(fc, 1);
				}
			}
		}
		
		FrequentConditionalCountComparator comparator = new FrequentConditionalCountComparator();
		PriorityQueue<FrequentConditionalStringVector> pq = new PriorityQueue<FrequentConditionalStringVector>(Config.maxFrequentConditionals, comparator);
		//get top freq items
		for (Map.Entry<FrequentConditionalStringVector, Integer> entry : conditionalCountMap.entrySet()) {
			FrequentConditionalStringVector fc = entry.getKey();
			fc.count = entry.getValue();
			if(pq.size() < Config.maxFrequentConditionals) {
				//just add it
				pq.add(fc);
			} else {
				//peek the lowest one, remove if small
				if(pq.peek().count < fc.count) {
					pq.remove();
					pq.add(fc);
				} 				
			}
		}
		//iterate over pq list and add into the treeset
		for(FrequentConditionalStringVector f : pq) {
			//System.out.print(f.index + "=" + f.count + ", ");
			frequentConditionals.add(f);
		}
		System.out.println();
		System.out.println("Frequent conditional size : " + frequentConditionals.size());
	}
	
	public static double getProbability(int y) {
		if(Config.vocabSamplingType.equals("unigram")) {
			return Corpus.vocabSampler.distribution.get(y);
		}
		else if(Config.vocabSamplingType.equals("uniform")) {
			return 1.0 / Config.VOCAB_SAMPLE_SIZE;
		}
		throw new RuntimeException("sampling type unrecognized : " + Config.vocabSamplingType);
	}
	
	public void setupSampler() {
		computeUnigramProbabilities();
		System.out.println("Vocab sample size: " + VOCAB_SAMPLE_SIZE);
		System.out.print("Setting up sampler...");
		vocabSampler = new DiscreteSampler(unigramProbability);
		System.out.println("Done");
	}
	
	public static int getRandomVocabItem() {
		return vocabSampler.sample();
		//return random.nextInt(corpusVocab.get(0).vocabSize);
	}
	
	public static TreeSet<Integer> getRandomVocabSet() {
		TreeSet<Integer> randomVocabSet = new TreeSet<Integer>();
		for(int i=0; i<VOCAB_SAMPLE_SIZE; i++) {
			randomVocabSet.add(getRandomVocabItem());
		}
		return randomVocabSet;
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
		
		// make all the states from the previous hierarchy available even if they are not observed in the data
		// will add one more to each of the vocabs' frequency
		// have to modify this if the number of states change in each hierarchy
		for(int z=1; z<oneTimeStepObsSize; z++) {
			for(int i=0; i<Config.numStates; i++) {
				corpusVocab.get(z).addItem("" + i);
			}
		}
		
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
	public void generateRandomTrainingEStepSample(int size, int iterCount) {
		
		trainInstanceEStepSampleList = new InstanceList();
		//add all
		if(trainInstanceList.size() <= size || size < 0) {
			trainInstanceEStepSampleList.addAll(trainInstanceList);
			trainInstanceEStepSampleList.numberOfTokens = trainInstanceList.numberOfTokens;
		} else {
			//sample sequentially or randomly
			if(Config.sampleSequential) {
				//if the training data is not shuffled yet, shuffle it
				if(trainInstanceListRandomized == null) {
					trainInstanceListRandomized = new InstanceList();
					ArrayList<Integer> randomInts = new ArrayList<Integer>();			
					for(int i=0; i<trainInstanceList.size(); i++) {
						randomInts.add(i);
					}
					Collections.shuffle(randomInts,Config.random);
					for(int i=0; i<trainInstanceList.size(); i++) {
						Instance instance = trainInstanceList.get(randomInts.get(i));
						trainInstanceListRandomized.add(instance);
						trainInstanceListRandomized.numberOfTokens += instance.T;
					}					
				}
				int startIndex = (iterCount * Config.sampleSizeEStep) % trainInstanceListRandomized.size();
				int index = startIndex;
				for(int i=0; i<size; i++) {
					Instance instance = trainInstanceListRandomized.get(index);
					trainInstanceEStepSampleList.add(instance);
					trainInstanceEStepSampleList.numberOfTokens += instance.T;
					index++;
					//index can get higher than the size of the training corpus
					index = index % trainInstanceListRandomized.size();
				}
			} else {
				//sample randomly
				ArrayList<Integer> randomInts = new ArrayList<Integer>();			
				for(int i=0; i<trainInstanceList.size(); i++) {
					randomInts.add(i);
				}
				Collections.shuffle(randomInts,Config.random);
				for(int i=0; i<size; i++) {
					Instance instance = trainInstanceList.get(randomInts.get(i));
					trainInstanceEStepSampleList.add(instance);
					trainInstanceEStepSampleList.numberOfTokens += instance.T;
				}
			}			
		}
	}	
	
	/*
	 * should be subset of E-step samples (because the posterior distribution is needed)
	 */
	public void generateRandomTrainingMStepSample(int size) {
		trainInstanceMStepSampleList = new InstanceList();
		if(trainInstanceEStepSampleList.size() <= size || size < 0) {
			trainInstanceMStepSampleList.addAll(trainInstanceEStepSampleList);
			trainInstanceMStepSampleList.numberOfTokens = trainInstanceEStepSampleList.numberOfTokens;
		} else {
			ArrayList<Integer> randomInts = new ArrayList<Integer>();			
			for(int i=0; i<trainInstanceEStepSampleList.size(); i++) {
				randomInts.add(i);
			}
			Collections.shuffle(randomInts,Config.random);
			for(int i=0; i<size; i++) {
				Instance instance = trainInstanceEStepSampleList.get(randomInts.get(i));
				trainInstanceMStepSampleList.add(instance);
				trainInstanceMStepSampleList.numberOfTokens += instance.T;
			}			
		}
	}
	
	/*
	 * does not make a clone, just the reference
	 */
	public void generateRandomTestSample(int size) {
		testInstanceSampleList = new InstanceList();
		if(testInstanceList.size() <= size || size < 0) {
			testInstanceSampleList.addAll(testInstanceList);
			testInstanceSampleList.numberOfTokens = testInstanceList.numberOfTokens;
		} else {
			ArrayList<Integer> randomInts = new ArrayList<Integer>();			
			for(int i=0; i<testInstanceList.size(); i++) {
				randomInts.add(i);
			}
			Collections.shuffle(randomInts,Config.random);
			for(int i=0; i<size; i++) {
				Instance instance = testInstanceList.get(randomInts.get(i));
				testInstanceSampleList.add(instance);
				testInstanceSampleList.numberOfTokens += instance.T;
			}			
		}
	}	
	
	/*
	 * does not make a clone, just the reference
	 */
	public void generateRandomDevSample(int size) {
		devInstanceSampleList = new InstanceList();
		if(devInstanceList.size() <= size || size < 0) {
			devInstanceSampleList.addAll(devInstanceList);
			devInstanceSampleList.numberOfTokens = devInstanceList.numberOfTokens;
		} else {
			ArrayList<Integer> randomInts = new ArrayList<Integer>();			
			for(int i=0; i<devInstanceList.size(); i++) {
				randomInts.add(i);
			}
			Collections.shuffle(randomInts,Config.random);
			for(int i=0; i<size; i++) {
				Instance instance = devInstanceList.get(randomInts.get(i));
				devInstanceSampleList.add(instance);
				devInstanceSampleList.numberOfTokens += instance.T;
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
	
	public void displayUnigramProb() {
		for(int i=0; i<unigramProbability.size(); i++) {
			System.out.format("%s --> %.4f\n", corpusVocab.get(0).indexToWord.get(i), unigramProbability.get(i));
		}
	}
	
	public static void main(String[] args) throws IOException {
		String inFile = "/home/anjan/workspace/HMM/data/simple_corpus_sorted.txt";
		int vocabThreshold = 1;
		Corpus c = new Corpus("\\s+", vocabThreshold);
		Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(inFile);
		c.readVocab(inFile);
		c.corpusVocab.get(0).debug();
		c.computeUnigramProbabilities();
		c.setupSampler();
		c.displayUnigramProb();
		
		System.out.println();
		for(int i=0; i<10; i++) {
			System.out.println(c.corpusVocab.get(0).indexToWord.get(c.getRandomVocabItem()));
		}
		
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
