package model.inference;

import java.util.Random;

import program.Main;

public class VariationalParamAlpha {
	double[] alpha;
	int T;
	
	public VariationalParamAlpha(int T) {
		this.T = T;
		alpha = new double[T];
	}
	
	public void initializeRandom() {
		Random r = new Random(Main.seed);
		for(int i=0; i<T; i++) {
			alpha[i] = r.nextDouble() + 1e-200;
		}
	}
}
