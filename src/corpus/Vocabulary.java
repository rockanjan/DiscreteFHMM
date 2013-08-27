package corpus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import cc.mallet.grmm.learning.ACRF.UnigramTemplate;

import util.SmoothWord;

public class Vocabulary {
	boolean debug = false;
	boolean smooth = true;
	boolean lower = true;
	public int vocabThreshold = 1;
	//index zero reserved for *unk* (low freq features)
	
	public int vocabReadIndex = 0;
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
			wordToIndex.put(word, vocabReadIndex);
			indexToWord.add(word);
			indexToFrequency.put(vocabReadIndex, 1);
			returnId = vocabReadIndex;
			vocabReadIndex++;
		}
		return returnId;
	}
	
	/* used when reading from dictionary */
	public int addItem(String word, int freq) {
		int returnId = -1;
		if(wordToIndex.containsKey(word)) {
			throw new RuntimeException(word + " found more than once in the dictionary");
		} else {
			wordToIndex.put(word, vocabReadIndex);
			indexToWord.add(word);
			indexToFrequency.put(vocabReadIndex, freq);
			returnId = vocabReadIndex;
			vocabReadIndex++;
		}
		return returnId;
	}
	
	//only called for the word vocab
	public void reduceVocab(Corpus c) {
		System.out.println("Reducing vocab");
		Map<String, Integer> wordToIndexNew = new HashMap<String, Integer>();
		ArrayList<String> indexToWordNew = new ArrayList<String>();
		Map<Integer, Integer> indexToFrequencyNew = new HashMap<Integer, Integer>();
		wordToIndexNew.put(UNKNOWN, 0);
		if(wordToIndex.containsKey(UNKNOWN)) {
			indexToFrequencyNew.put(0, indexToFrequency.get(0)); //TODO: decide if this matters
		} else {
			indexToFrequencyNew.put(0, 0);
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
	
	public void writeDictionary(String filename) {
		try{
			PrintWriter pw = new PrintWriter(filename);
			//write vocabSize
			pw.println(this.vocabSize);
			for(int i=0; i<indexToWord.size(); i++) {
				pw.println(indexToWord.get(i) + " " + indexToFrequency.get(i));
			}
			pw.close();
		}
		catch(IOException ioe) {
			ioe.printStackTrace();
			System.exit(-1);
		}
	}
	
	//reads from the dictionary
	public void readDictionary(String filename) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		wordToIndex.clear();
		String line = null;
		try {
			line = br.readLine().trim();
			vocabSize = Integer.parseInt(line);
			while( (line = br.readLine()) != null) {
				line = line.trim();
				if(line.isEmpty()) {
					continue;
				}
				String[] splitted = line.split("\\s+");
				String word = splitted[0];
				int freq = Integer.parseInt(splitted[1]);
				addItem(word, freq);
			}
			System.out.println("Dictionary Loaded Vocab Size: " + wordToIndex.size());
		} catch (IOException e) {
			e.printStackTrace();
			System.err.println("error reading dictionary file");
		}
		if(vocabSize != wordToIndex.size()) {
			System.out.println("dictionary file corrputed: header size and the vocab size do not match");
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
			sb.append("\t" + i + " --> " + "\t" + indexToWord.get(i) + " --> " + indexToFrequency.get(i));
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
