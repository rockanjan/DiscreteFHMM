package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import util.SmoothWord;

public class Vocabulary {
	boolean debug = true;
	boolean smooth = true;
	boolean lower = true;
	public int vocabThreshold = 1;
	//index zero reserved for *unk* (low freq features)
	
	public int index = -1;
	public int vocabSize = -1;
	public static String UNKNOWN = "*unk*";
	public Map<String, Integer> wordToIndex = new HashMap<String, Integer>();
	public ArrayList<String> indexToWord = new ArrayList<String>();
	public Map<Integer, Integer> indexToFrequency = new HashMap<Integer, Integer>();
	
	public int addItem(String word) {
		int returnId = -1;
		if(wordToIndex.containsKey(word)) {
			int wordIndex = wordToIndex.get(word);
			int oldFreq = indexToFrequency.get(wordIndex);
			indexToFrequency.put(wordIndex, oldFreq + 1);
			returnId = wordIndex;
		} else {
			wordToIndex.put(word, index);
			indexToWord.add(word);
			indexToFrequency.put(index, 1);
			returnId = index;
			index++;
		}
		return returnId;
	}
	
	public void reduceVocab(Corpus c) {
		System.out.println("Reducing vocab");
		Map<String, Integer> wordToIndexNew = new HashMap<String, Integer>();
		ArrayList<String> indexToWordNew = new ArrayList<String>();
		Map<Integer, Integer> indexToFrequencyNew = new HashMap<Integer, Integer>();
		wordToIndexNew.put(UNKNOWN, 0);
		if(wordToIndex.containsKey(UNKNOWN)) {
			indexToFrequencyNew.put(0, indexToFrequency.get(0)); //TODO: decide if this matters
		} else {
			indexToFrequencyNew.put(0, 0); //TODO: decide if this matters
		}
		indexToWordNew.add(UNKNOWN);
		
		int featureIndex = 1;
		for(int i=1; i<indexToWord.size(); i++) {
			if(indexToFrequency.get(i) > vocabThreshold) {
				wordToIndexNew.put(indexToWord.get(i), featureIndex);
				indexToWordNew.add(indexToWord.get(i));
				indexToFrequencyNew.put(featureIndex, indexToFrequency.get(i));
				featureIndex = featureIndex + 1;
			} else {
				indexToFrequencyNew.put(0, indexToFrequencyNew.get(0) + indexToFrequency.get(i));
			}
		}
		indexToWord = null; indexToFrequency = null; wordToIndex = null;
		indexToWord = indexToWordNew;
		indexToFrequency = indexToFrequencyNew;
		wordToIndex = wordToIndexNew;
		vocabSize = wordToIndex.size();
		//System.out.println("New vocab size : " + vocabSize);
		
	}
	
	//reads from the dictionary
	public void readVocabFromDictionary(String filename) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		String line = null;
		try {
			line = br.readLine().trim();
			vocabSize = Integer.parseInt(line);
			while( (line = br.readLine()) != null) {
				line = line.trim();
				if(line.isEmpty()) {
					continue;
				}
				addItem(line);
			}
			System.out.println("Loaded Vocab Size : " + wordToIndex.size());
		} catch (IOException e) {
			e.printStackTrace();
			System.err.println("error reading vocab file");
		}
		if(vocabSize != wordToIndex.size()) {
			System.out.println("Vocab file corrputed: header size and the vocab size do not match");
			System.exit(-1);
		}
	}
	
	public void debug() {
		StringBuffer sb = new StringBuffer();
		sb.append("DEBUG: Corpus\n");
		sb.append("=============\n");
		sb.append("vocab size : " + vocabSize);
		sb.append("\nvocab frequency: \n");
		for (int i = 0; i < vocabSize; i++) {
			sb.append("\t" + indexToWord.get(i) + " --> "
					+ indexToFrequency.get(i));
			sb.append("\n");
		}
		System.out.println(sb.toString());
	}
	
	public int getIndex(String word) {
		if(wordToIndex.containsKey(word)) {
			return wordToIndex.get(word);
		} else {
			//word not found in vocab
			if(debug) {
				System.out.println(word + " not found in vocab");
			}
			return 0; //unknown id
		}
	}
	
}
