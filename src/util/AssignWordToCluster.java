package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import cc.mallet.util.CommandOption.Set;

/*
 * Reads an HMM decoded file and assigns each word to each (non overlapping) cluster
 * approach: assign most frequent cluster for the word
 */
public class AssignWordToCluster {
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		String inFile="brown_train.txt.decoded";
		int WORD_COL = 2;
		int HMM_COL = 1;
		
		//BEGIN: find number of clusters (do it easy way, find the highest index)		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		String line;
		int numClusters = 0;
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
		
		//BEGIN: collect how many times a word appears with a HMM state
		Map<String, ClusterCount> wordToClusterCounts; //stores the count of clusters for each word
		wordToClusterCounts = new HashMap<String, ClusterCount>();
		br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		//populate word to cluster counts
		while ( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty() ) {
				String[] splitted = line.split("(\\s+|\\t+)");
				int hmm = Integer.parseInt(splitted[HMM_COL-1]);
				String word = splitted[WORD_COL-1];
				if(! wordToClusterCounts.containsKey(word)) {
					ClusterCount cc = new ClusterCount(numClusters);
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
		
		PrintWriter pw;
		
		double[] numWordsInCluster;
		
		//BEGIN: max assignment
		numWordsInCluster = new double[numClusters];
		int total=0;
		pw = new PrintWriter(inFile + ".cluster.max");
		for(String word : wordToClusterCounts.keySet()) {
			ClusterCount cc = wordToClusterCounts.get(word);
			int maxCluster = cc.getMaxIndex();
			pw.println(maxCluster + " " + word);
			numWordsInCluster[maxCluster]++;
			total++;
		}
		pw.close();
		for(int i=0; i<numClusters; i++) {
			numWordsInCluster[i] /= total;
		}
		System.out.println("Entropy for max assignment: " + MathUtils.getEntropy(numWordsInCluster));
		//END: max assignment
		
		//BEGIN: prob distribution assignment
		Random r = new Random();
		numWordsInCluster = new double[numClusters];
		total = 0;
		pw = new PrintWriter(inFile + ".cluster.dist");
		for(String word : wordToClusterCounts.keySet()) {
			ClusterCount cc = wordToClusterCounts.get(word);
			cc.setupSampler();
			int sampledCluster = cc.sampler.sample(r);
			pw.println(sampledCluster + " " + word);
			numWordsInCluster[sampledCluster]++;
			total++;
		}
		pw.close();
		for(int i=0; i<numClusters; i++) {
			numWordsInCluster[i] /= total;
		}
		System.out.println("Entropy for dist assignment: " + MathUtils.getEntropy(numWordsInCluster));
		//END: prob distribution assignment
	}	
}
