package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.TreeSet;

import corpus.FrequentConditionalCountComparator;
import corpus.FrequentConditionalStringVector;
import corpus.VocabItemProbComparator;
import corpus.VocabItemProbability;
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
		
		System.out.println("__________");
		VocabItemProbComparator comparator = new VocabItemProbComparator();
		PriorityQueue<VocabItemProbability> pq = new PriorityQueue<VocabItemProbability>(10, comparator);
		VocabItemProbability v1 = new VocabItemProbability(10, 0.5);
		VocabItemProbability v2 = new VocabItemProbability(20, 0.7);
		VocabItemProbability v3 = new VocabItemProbability(15, 0.12);
		VocabItemProbability v4 = new VocabItemProbability(30, 0.3);
		VocabItemProbability v5 = new VocabItemProbability(40, 0.2);
		pq.add(v1);
		pq.add(v2);
		pq.add(v3);
		pq.add(v4);
		pq.add(v5);
		
		System.out.println("Traversal");
		//just traversal, we don't get specific order
		for(VocabItemProbability v : pq) {
			System.out.println(v);
		}
		
		
		System.out.println("removal");
		//with removal, we get particular order
		while (pq.size() != 0)
        {
            System.out.println(pq.remove());
        }
		/*
		for(VocabItemProbability v : pq) {
			System.out.println(v);
		}
		*/
		
		FrequentConditionalCountComparator comp = new FrequentConditionalCountComparator();
		PriorityQueue<FrequentConditionalStringVector> pqString = new PriorityQueue<FrequentConditionalStringVector>(10, comp);
		pqString.add(new FrequentConditionalStringVector("0", new double[1]));
		pqString.add(new FrequentConditionalStringVector("1", new double[1]));
		
		TreeSet<FrequentConditionalStringVector> treeSet = new TreeSet<FrequentConditionalStringVector>(comp);
		System.out.println("PQ size : " + pqString.size());
		for(FrequentConditionalStringVector e : pqString) {
			treeSet.add(e);
		}
		System.out.println("Treeset size : " + treeSet.size());
		
		
	}	
}

