package corpus;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import util.SmoothWord;

public class Vocabulary {
	boolean debug = false;
	boolean smooth = true;
	boolean lower = true;
	public int vocabThreshold = 1;
	//index zero reserved for *unk* (low freq features)
	
	private int index = 0;
	public int vocabSize = -1;
	public String UNKNOWN = "*unk*";
	public Map<String, Integer> wordToIndex = new HashMap<String, Integer>();
	public ArrayList<String> indexToWord = new ArrayList<String>();
	public Map<Integer, Integer> indexToFrequency = new HashMap<Integer, Integer>();
	
	private int addItem(String word) {
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
	
	public void readVocabFromCorpus(Corpus c, String filename) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF8"));
		String line = null;
		wordToIndex.put(UNKNOWN, 0);
		indexToFrequency.put(0, 0);
		indexToWord.add(UNKNOWN);
		index = 1; //indexToFrequency should start new index from 1
		while( (line = br.readLine()) != null) {
			//System.out.println(line);
			line = line.trim();
			if(! line.isEmpty()) {
				String words[] = line.split(c.delimiter);
				for(int i=0; i<words.length; i++) {
					String word = words[i];
					if(lower) {
						word = word.toLowerCase();
					}
					if(smooth) {
						word = SmoothWord.smooth(word);
					}
					
					int wordId = addItem(word);					
				}
			}
		}
		vocabSize = wordToIndex.size();
		System.out.println("Vocab Size before reduction including UNKNOWN : " + vocabSize);
		if(debug) {
			c.debug();
		}
		reduceVocab(c);
		System.out.println("Vocab Size after reduction including UNKNOWN : " + vocabSize);
		if(debug) {
			c.debug();
		}
		br.close();		
	}
	
	private void reduceVocab(Corpus c) {
		System.out.println("Reducing vocab");
		Map<String, Integer> wordToIndexNew = new HashMap<String, Integer>();
		ArrayList<String> indexToWordNew = new ArrayList<String>();
		Map<Integer, Integer> indexToFrequencyNew = new HashMap<Integer, Integer>();
		wordToIndexNew.put(UNKNOWN, 0);
		indexToFrequencyNew.put(0, indexToFrequency.get(0)); //keep the UNK no matter what
		indexToWordNew.add(UNKNOWN);
		int featureIndex = 1;
		int unkCount = 0;
		for(int i=1; i<indexToWord.size(); i++) {
			if(indexToFrequency.get(i) > vocabThreshold) {
				wordToIndexNew.put(indexToWord.get(i), featureIndex);
				indexToWordNew.add(indexToWord.get(i));
				indexToFrequencyNew.put(featureIndex, indexToFrequency.get(i));
				featureIndex = featureIndex + 1;
			}
			else {
				unkCount += indexToFrequency.get(i);
			}
		}
		indexToFrequencyNew.put(0, unkCount);
		indexToWord = null; indexToFrequency = null; wordToIndex = null;
		indexToWord = indexToWordNew;
		indexToFrequency = indexToFrequencyNew;
		wordToIndex = wordToIndexNew;
		vocabSize = wordToIndex.size();
		//System.out.println("New vocab size : " + vocabSize);
		
	}
	
	//reads from the dictionary
	public void readVocabFromDictionary(String filename) throws UnsupportedEncodingException {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF8"));
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
