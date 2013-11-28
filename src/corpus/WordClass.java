package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
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
					System.err.println("WARN: number of OOV words found = " + oovCount + " word = " + word);
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
		if(oovCount > 0) {
			System.exit(-1);
		}
		System.out.println("Word index to Cluster index size = " + wordIndexToClusterIndex.size());
		//view one cluster
		System.out.println("Viewing one cluster");
		for( int wordIndex : clusterIndexToWordIndices.get(0) ){
			System.out.print(Corpus.corpusVocab.get(0).indexToWord.get(wordIndex) + " ");
		}
		System.out.println();
		br.close();
		
		//save cluster index file
		PrintWriter pw = new PrintWriter(Config.baseDirModel + "cluster_vocabs.txt");
		for(int clusterIndex : clusterIndexToWordIndices.keySet()) {
			pw.print(clusterIndex + ":");
			Set<Integer> wordIndices = clusterIndexToWordIndices.get(clusterIndex);
			for(int wordIndex : wordIndices) {
				pw.print(wordIndex + " ");
			}
			pw.println();
			pw.flush();
		}
		pw.close();
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
		br.close();
		WordClass.numClusters = clusterNames.size();
		System.out.println("Total clusters : " + clusterNames.size());
	}
	
	public static void populateClustersFromSavedFile() throws IOException {
		//only need to populate wordIndexToClusterIndex and clusterIndexToWordIndices
		wordIndexToClusterIndex = new HashMap<Integer, Integer>();
		clusterIndexToWordIndices = new HashMap<Integer, Set<Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(Config.baseDirModel + "cluster_vocabs.txt"));
		String line;
		while((line = br.readLine()) != null) {
			line = line.trim();
			if(line.isEmpty()) continue;
			String[] splitted = line.split(":");
			int clusterId = Integer.parseInt(splitted[0]);
			String[] splittedWordIndices = splitted[1].split(" ");
			Set<Integer> wordsInCluster = new HashSet<Integer>(); 
			for(String wordIndexString : splittedWordIndices) {
				int wordId = Integer.parseInt(wordIndexString);
				wordsInCluster.add(wordId);
				wordIndexToClusterIndex.put(wordId, clusterId);
			}
			clusterIndexToWordIndices.put(clusterId, wordsInCluster);
		}
		numClusters = clusterIndexToWordIndices.size();
		System.out.println("Total clusters loaded: " + numClusters);
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
 