package test;

import java.util.ArrayList;

import util.Timing;
import corpus.Vocabulary;

public class Test {
	
	public static double exp(double val) {
	    final long tmp = (long) (1512775 * val + (1072693248 - 60801));
	    return Double.longBitsToDouble(tmp << 32);
	}

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
		
		double a = .002342323423423425;
		Timing t = new Timing();
		t.start();
		for(int i=0; i<100000000; i++) {
			Math.exp(a);
		}
		System.out.println(Math.exp(a));
		System.out.println(t.stopMilliseconds());
		t.start();
		
		for(int i=0; i<100000000; i++) {
			exp(a);
		}
		System.out.println(exp(a));
		System.out.println(" " + t.stopMilliseconds());
	}
}
