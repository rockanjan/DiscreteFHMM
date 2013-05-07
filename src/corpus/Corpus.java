package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import util.SmoothWord;

public class Corpus {
	public static String delimiter = "\\s+";
	public static String obsDelimiter = "\\|"; //separator of multiple observation elements at one timestep
	public static int oneTimeStepObsSize = -1;
	public InstanceList trainInstanceList = new InstanceList();
	// testInstanceList can be empty
	public InstanceList testInstanceList;

	public ArrayList<Vocabulary> corpusVocab;

	int vocabThreshold;
	
	public int totalWords; 

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
					System.err
							.println("Could not read from train file, line = "
									+ line);
				}
			}
		}
		System.out.println("Train Instances: " + trainInstanceList.size());
		System.out.println("Train token count: " + totalWords);
		System.out.println("Train unknown count : " + totalUnknown);
		br.close();
	}

	public void readVocab(String inFile) throws IOException {
		corpusVocab = new ArrayList<Vocabulary>();
		Vocabulary v = new Vocabulary(); //base vocabulary
		v.index = 1; //zero reserved for *unk*
		v.vocabThreshold = vocabThreshold;
		corpusVocab.add(v);
		
		//hmm states as vocabs
		for(int i=1; i<oneTimeStepObsSize; i++) {
			Vocabulary vHmmStates = new Vocabulary();
			vHmmStates.vocabThreshold = 0;
			vHmmStates.index = 0;
			corpusVocab.add(vHmmStates);
		}
		readVocabFromCorpus(inFile);
	}
	
	public void readVocabFromCorpus(String filename) throws IOException {
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
	
	public void saveVocabFile(String filename) {
		PrintWriter dictionaryWriter;
		try {
			dictionaryWriter = new PrintWriter(filename);
			int V = corpusVocab.vocabSize;
			dictionaryWriter.println(V);
			for (int v = 0; v < V; v++) {
				dictionaryWriter.println(corpusVocab.indexToWord.get(v));
				dictionaryWriter.flush();
			}
			dictionaryWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}	
	*/
	
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
		//String inFile = "/home/anjan/workspace/HMM/out/decoded/test.decoded.txt";
		//String inFile = "/home/anjan/workspace/HMM/data/test.txt.SPL";
		
		//String inFile = "/home/anjan/workspace/HMM/data/train.txt.small.SPL";
		String inFile = "/home/anjan/workspace/HMM/out/decoded/train.txt.small.SPL";
		
		int vocabThreshold = 1;
		Corpus c = new Corpus("\\s+", vocabThreshold);
		Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(inFile);
		c.readVocab(inFile);
	}
}
