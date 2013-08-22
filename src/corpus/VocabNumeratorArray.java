package corpus;

import java.util.PriorityQueue;

public class VocabNumeratorArray {
	//don't have to actually store this array 
	//because only top few will be used which are stored in the PQ
	public double[] array;
	public double normalizer;
	public PriorityQueue<VocabItemProbability> topProbs;
	
	//conditional if processed in gradient update
	public boolean processed = false;
	//how many conditionals for this
	public int count; 
	
	public VocabNumeratorArray(int size) {
		array = new double[size];
	}
}
