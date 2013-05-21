package test;

import java.util.ArrayList;

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
	}
}
