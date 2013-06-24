package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

import corpus.Vocabulary;

public class Test {
	private int id;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		ArrayList<Vocabulary> vocabList = new ArrayList<Vocabulary>();
		for(int i=0; i<3; i++) {
			Vocabulary v = new Vocabulary();
			vocabList.add(v);
			v = null;
		}
		System.out.println(vocabList.get(2));
		int x = 10;
		int y = 9;
		double z = x/y * 1.0; //first divides integers
		//double z = 1.0 * x/y;
		System.out.println(z);
		
		final Integer[] idx = { 0, 1, 2, 3 };
		final float[] data = { 1.7f, -0.3f,  2.1f,  2.1f };

		Arrays.sort(idx, new Comparator<Integer>() {
		    @Override public int compare(final Integer o1, final Integer o2) {
		        return Float.compare(data[o2], data[o1]);
		    }
		});
		
		for(int i=0; i<idx.length; i++) {
			System.out.println(idx[i] + "\t" + data[idx[i]]);
		}	
		
	}	
}
