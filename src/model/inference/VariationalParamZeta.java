package model.inference;

import java.util.Random;

public class VariationalParamZeta {
	int V; //vocab size
	double zeta[];
	public VariationalParamZeta(int V) {
		this.V = V;
		zeta = new double[V];
	}
	
	public void initializeRandom() {
		Random r = new Random();
		for(int i=0; i<zeta.length; i++) {
			zeta[i] = r.nextDouble();
		}
	}

	/*
	 * returns the value for the function lambda(zeta_k);
	 */
	public double lamdaZeta(int index) {
		return 0.5/zeta[index] * (1/(1+Math.exp(-zeta[index])) - 0.5);
	}
}
