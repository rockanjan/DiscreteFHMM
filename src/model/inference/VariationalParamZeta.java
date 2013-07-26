package model.inference;

import java.util.Random;

import util.MathUtils;

public class VariationalParamZeta {
	int V; //vocab size
	double zeta[];
	public VariationalParamZeta(int V) {
		this.V = V;
		zeta = new double[V];
	}
	
	public void initializeRandom() {
		Random r = new Random();
		double small = 1e-100;
		for(int i=0; i<zeta.length; i++) {
			zeta[i] = r.nextDouble() + small;
		}
	}

	/*
	 * returns the value for the function lambda(zeta_k);
	 */
	public double lamdaZeta(int index) {
		double value = 0.5/zeta[index] * (1/(1+Math.exp(-zeta[index])) - 0.5);
		try{
			MathUtils.check(value);
		} catch (Exception e) {
			System.out.println("zeta : " + zeta[index]);
		}
		return value;
	}
}
