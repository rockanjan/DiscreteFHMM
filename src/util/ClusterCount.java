package util;

public class ClusterCount {
	String word;
	public int[] counts;
	public ClusterCount(int size) {
		counts = new int[size];
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
