package util;

import java.util.Random;

import corpus.DiscreteSampler;

/*
 * Data structure for Word and cluster
 */
public class WordClusterDS {
	String word;
	public int[] counts;
	double[] distribution;
	DiscreteSampler sampler;
	public double entropy; //entropy of the distribution of clusters in the word
	public int assignedCluster;
	
	public WordClusterDS(int size) {
		counts = new int[size];
	}
	
	public void setupSampler() {
		int sum = 0;
		for(int i=0; i<counts.length; i++) {
			sum += counts[i];
		}
		distribution = new double[counts.length];
		for(int i=0; i<counts.length; i++) {
			distribution[i] = 1.0 * counts[i] / sum;
		}
		sampler = new DiscreteSampler(distribution);
		
	}
	
	public int sampleNewCluster() {
		return sampler.sample(new Random());
	}
	
	public int getMaxIndex() {
		int maxIndex = -1;
		int maxCount = -1;
		for(int i=0; i<counts.length; i++) {
			int c = counts[i];
			if(c > maxCount ) {
				maxCount = c;
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
}
