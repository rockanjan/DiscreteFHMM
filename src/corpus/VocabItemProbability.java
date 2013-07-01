package corpus;

import java.util.PriorityQueue;

public class VocabItemProbability {
	public int index;
	public double prob;
	
	public VocabItemProbability(int i, double p) {
		index = i;
		prob = p;
	}
	
	@Override
	public String toString() {
		return index + " --> " + prob;
	}
	
}
