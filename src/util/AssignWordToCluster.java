package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/*
 * Reads an HMM decoded file and assigns each word to each (non overlapping) cluster
 * approach: assign most frequent cluster for the word
 */
public class AssignWordToCluster {
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		String inFile="brown_train.txt.decoded";
		Map<String, ClusterCount> wordToClusterCounts; //stores the count of clusters for each word
		wordToClusterCounts = new HashMap<String, ClusterCount>();
		int clusterSize = findNumberOfClusters();
		
		int WORD_COL = 2;
		int HMM_COL = 1;
		//BufferedReader br = new BufferedReader(new FileReader(inFile));
		BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		String line;
		//populate word to cluster counts
		while ( (line = br.readLine()) != null) {
			line = line.trim();
			if(! line.isEmpty() ) {
				String[] splitted = line.split("(\\s+|\\t+)");
				int hmm = Integer.parseInt(splitted[HMM_COL-1]);
				String word = splitted[WORD_COL-1];
				if(! wordToClusterCounts.containsKey(word)) {
					ClusterCount cc = new ClusterCount(clusterSize);
					cc.word = word;
					cc.counts[hmm] = 1;
					wordToClusterCounts.put(word, cc);
				} else {
					wordToClusterCounts.get(word).counts[hmm]++;
				}
			}
		}
		br.close();
		PrintWriter pw = new PrintWriter(inFile + ".cluster");
		for(String word : wordToClusterCounts.keySet()) {
			ClusterCount cc = wordToClusterCounts.get(word);
			int maxCluster = cc.getMaxIndex();
			pw.println(maxCluster + " " + word);		
		}
		pw.close();
	}
	
	private static int findNumberOfClusters() {
		//TODO
		return 100;
	}
}
