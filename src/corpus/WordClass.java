package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import util.SmoothWord;
import config.Config;

public class WordClass {
	public static Map<Integer, Integer> wordIndexToClusterIndex;
	public static Map<Integer, Set<Integer>> clusterIndexToWordIndices;
	public static List<String> clusterNames;
	public static Map<String, Integer> clusterNameToIndex;
	
	public static int numClusters = -1;
	public static void populateClassInfo() throws IOException {
		if(Corpus.corpusVocab.get(0) == null) {
			throw new RuntimeException("Corpus vocab should be initialized before populating word class info");
		}
		populateClusters(); //get dictionary of cluster and create indices
		
		wordIndexToClusterIndex = new HashMap<Integer, Integer>();
		clusterIndexToWordIndices = new HashMap<Integer, Set<Integer>>();
		String file = Config.wordClusterFile;
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int oovCount = 0;
		while( (line = br.readLine()) != null) {
			String[] splitted = line.trim().split("\\s+");
			String cluster = splitted[0];
			int clusterIndex = clusterNameToIndex.get(cluster);
			
			String word = splitted[1];
			word = SmoothWord.smooth(word);
			int wordIndex = Corpus.corpusVocab.get(0).getIndex(word);
			if(wordIndex == 0) {
				oovCount++;
				if(oovCount > 1) {
					System.err.println("WARN: number of OOV words found = " + oovCount);
				}
			}
			wordIndexToClusterIndex.put(wordIndex, clusterIndex);
			if(! clusterIndexToWordIndices.containsKey(clusterIndex)) {
				HashSet<Integer> temp = new HashSet<Integer>();
				temp.add(wordIndex);
				clusterIndexToWordIndices.put(clusterIndex, temp);
			} else {
				clusterIndexToWordIndices.get(clusterIndex).add(wordIndex);
			}
		}
		System.out.println("Word index to Cluster index size = " + wordIndexToClusterIndex.size());
		//view one cluster
		System.out.println("Viewing one cluster");
		for( int wordIndex : clusterIndexToWordIndices.get(0) ){
			System.out.print(Corpus.corpusVocab.get(0).indexToWord.get(wordIndex) + " ");
		}
		System.out.println();
		br.close();
	}	
	
	private static void populateClusters() throws IOException {
		String file = Config.wordClusterFile;
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int index = 0;
		clusterNames = new ArrayList<String>();
		clusterNameToIndex = new HashMap<String, Integer>();
		while( (line = br.readLine()) != null) {
			line.trim();
			if(line.isEmpty()) continue;
			String[] splitted = line.split("\\s+");
			String cluster = splitted[0];
			if(! clusterNameToIndex.containsKey(cluster)) {
				clusterNameToIndex.put(cluster, index++);
				clusterNames.add(cluster);
			}
		}
		WordClass.numClusters = clusterNames.size();
		System.out.println("Total clusters : " + clusterNames.size());
		br.close();
	}
	
	public static void main(String[] args) throws IOException {
		String inFile = "data/brown_train.txt";
		int vocabThreshold = 3;
		Corpus c = new Corpus("\\s+", vocabThreshold);
		Corpus.oneTimeStepObsSize = Corpus.findOneTimeStepObsSize(inFile);
		c.readVocab(inFile);
		//Config.wordClusterFile = "brown-brown-cluster.txt";
		WordClass.populateClassInfo();
		
	}
}
