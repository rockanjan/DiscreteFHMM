package test;

import java.util.ArrayList;

import util.MathUtils;
import util.Timing;
import corpus.Vocabulary;

public class Test {
	
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
		
	}
}
