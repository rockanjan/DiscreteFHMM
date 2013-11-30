package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import cc.mallet.util.CommandOption.Set;

/*
 * Reads an HMM decoded file and assigns each word to each (non overlapping) cluster
 * approach: assign most frequent cluster for the word
 */
public class AssignWordToCluster {
	static String inFile="brown_train.txt.decoded";
	static int WORD_COL = 2;
	static int HMM_COL = 1;
	static int MAX_WORDS_PER_CLUSTER = 1000;
	
	static int numClusters;
	static Map<String, WordClusterDS> wordToClusterCounts; //stores the count of clusters for each word
	//cluster to list of words 
	static Map<Integer, List<WordClusterDS>> clusterToWords;
	
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		findNumClusters();
		populateWordToClusterCount();
		setRandomAssignment();
		printWordCluster(inFile + ".cluster.random");
		//setMaxAssignment();
		//printWordCluster(inFile + ".cluster.max");
		//split clusters with many words (based on the entropy)
		//splitClusters();
		//printWordCluster(inFile + ".cluster.split");
		
	}	
	
	public static class EntropyComparator implements Comparator<WordClusterDS> {

		@Override
		public int compare(WordClusterDS o1, WordClusterDS o2) {
			if(o1.entropy > o2.entropy) {
				return 1;
			} else if(o1.entropy < o2.entropy) {
				return -1;
			}
			return 0;
		}
		
	}
	
	public static void splitClusters() {
		//populate clusterToWords based on previous assignment
		clusterToWords = new HashMap<Integer, List<WordClusterDS>>();
		for(String word : wordToClusterCounts.keySet()) {
			WordClusterDS wc = wordToClusterCounts.get(word);
			int clusterId = wc.assignedCluster;
			if(clusterToWords.containsKey(clusterId)) {
				clusterToWords.get(clusterId).add(wc);
			} else {
				List<WordClusterDS> listWc = new ArrayList<WordClusterDS>();
				listWc.add(wc);
				clusterToWords.put(clusterId, listWc);
			}
		}
		EntropyComparator comparator = new EntropyComparator();
		//first, for each cluster, sort the words based on the entropy of the cluster distribution
		for(Integer cluster : clusterToWords.keySet()) {
			List<WordClusterDS> wordList = clusterToWords.get(cluster);
			if(wordList.size() > MAX_WORDS_PER_CLUSTER) {
				System.out.println("clusterToWordsSize = " + wordList.size());
				for(WordClusterDS wc : wordList) {
					wc.setupSampler(); //computes the distribution
					wc.entropy = MathUtils.getEntropy(wc.distribution);
				}
				//sort them
				Collections.sort(wordList, comparator);
				int divideSize = wordList.size() / MAX_WORDS_PER_CLUSTER;
				//leave the first MAX_WORDS in the original cluster
				for(int i=1; i<=divideSize; i++) {
					int startIndex = i*MAX_WORDS_PER_CLUSTER;
					int endIndex = startIndex + MAX_WORDS_PER_CLUSTER;
					if(endIndex > wordList.size()) {
						endIndex = wordList.size();
					}
					if(startIndex < wordList.size()) {
						numClusters++; //increase the cluster index
						for(int j=startIndex; j<endIndex; j++) {
							wordList.get(j).assignedCluster = numClusters;
						}
					}
				}
			}
		}
		System.out.println("Num clusters after splitting = " + numClusters);
	}
	
	public static void printWordCluster(String outFile) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(outFile);
		for(String word : wordToClusterCounts.keySet()) {
		   	WordClusterDS wc = wordToClusterCounts.get(word);
		   	pw.println(wc.assignedCluster + " " + word);
		}
		pw.close();
	}
	
	public static void setRandomAssignment() {
		Random r = new Random();
		for(String word : wordToClusterCounts.keySet()) {
			WordClusterDS cc = wordToClusterCounts.get(word);
			int randomCluster = r.nextInt(numClusters);
			cc.assignedCluster = randomCluster;
		}
	}
	
	public static void setMaxAssignment() {
		for(String word : wordToClusterCounts.keySet()) {
			WordClusterDS cc = wordToClusterCounts.get(word);
			int maxCluster = cc.getMaxIndex();
			cc.assignedCluster = maxCluster;
		}
	}
	
	public static void setDistributionAssignment() {
		//BEGIN: prob distribution assignment
		Random r = new Random();
		double[] numWordsInCluster = new double[numClusters];
		double total = 0;
		for(String word : wordToClusterCounts.keySet()) {
			WordClusterDS cc = wordToClusterCounts.get(word);
			cc.setupSampler();
			int sampledCluster = cc.sampler.sample(r);
			cc.assignedCluster = sampledCluster;
			numWordsInCluster[sampledCluster]++;
			total++;
		}
		for(int i=0; i<numClusters; i++) {
			numWordsInCluster[i] /= total;
		}
		System.out.println("Entropy for dist assignment: " + MathUtils.getEntropy(numWordsInCluster));
		//END: prob distribution assignment
	}
	
	public static void populateWordToClusterCount() throws NumberFormatException, IOException {
		//BEGIN: collect how many times a word appears with a HMM state
		wordToClusterCounts = new HashMap<String, WordClusterDS>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		//populate word to cluster counts
		String line;
		while ( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty() ) {
				String[] splitted = line.split("(\\s+|\\t+)");
				int hmm = Integer.parseInt(splitted[HMM_COL-1]);
				String word = splitted[WORD_COL-1];
				if(! wordToClusterCounts.containsKey(word)) {
					WordClusterDS cc = new WordClusterDS(numClusters);
					cc.word = word;
					cc.counts[hmm] = 1;
					wordToClusterCounts.put(word, cc);
				} else {
					wordToClusterCounts.get(word).counts[hmm]++;
				}
			}
		}
		br.close();
		//END: collect how many times a word appears with a HMM state
	}
	
	
	public static void findNumClusters() throws NumberFormatException, IOException {
		//BEGIN: find number of clusters (do it easy way, find the highest index)		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		String line;
		numClusters = 0;
		while ( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty() ) {
				String[] splitted = line.split("(\\s+|\\t+)");
				int hmm = Integer.parseInt(splitted[HMM_COL-1]);
				if(hmm > numClusters) {
					numClusters = hmm;
				}
			}
		}
		br.close();
		//finally add one
		System.out.println("Max HMM index found = " + numClusters);
		++numClusters;
		System.out.println("Taking total clusters to be " + numClusters);
		//END: find number of clusters
	}
	
	
}
