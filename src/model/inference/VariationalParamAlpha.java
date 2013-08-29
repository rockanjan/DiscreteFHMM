package model.inference;

import config.Config;

public class VariationalParamAlpha {
	double[] alpha;
	int T;
	
	public VariationalParamAlpha(int T) {
		this.T = T;
		alpha = new double[T];
	}
	
	public void initializeRandom() {
		for(int i=0; i<T; i++) {
			alpha[i] = Config.random.nextDouble() + 1e-200;
		}
	}
	
	public void initializeUniform(double value) {
		for(int i=0; i<T; i++) {
			alpha[i] = value;
		}
	}
}
