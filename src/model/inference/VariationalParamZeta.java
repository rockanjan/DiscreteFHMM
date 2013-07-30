package model.inference;

import java.util.Random;

import util.MathUtils;

public class VariationalParamZeta {
	int V; //vocab size
	double zeta[];
	double lambdaCache[];
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
		if(lambdaCache == null) {
			createLambdaCache();
		}
		return lambdaCache[index];
	}
	
	public void createLambdaCache() {
		lambdaCache = new double[V];
		//create cache
		for(int index=0; index<V; index++) {
			double value = 0.5/zeta[index] * (1/(1+Math.exp(-zeta[index])) - 0.5);
			try{
				MathUtils.check(value);
			} catch (Exception e) {
				System.err.println("zeta : " + zeta[index]);
			}
			lambdaCache[index] = value;
		}
	}
	
	public void clearLambdaCache() {
		lambdaCache = null;
	}
}
