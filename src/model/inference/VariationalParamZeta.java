package model.inference;

import java.util.Random;

import util.MathUtils;

public class VariationalParamZeta {
	int V; //vocab size
	int T; //timestep size
	double zeta[][];
	public VariationalParamZeta(int V, int T) {
		this.V = V;
		this.T = T;
		zeta = new double[V][T];
	}
	
	public void initializeRandom() {
		Random r = new Random();
		double small = 1e-100;
		for(int i=0; i<zeta.length; i++) {
			for(int t=0; t<T; t++) {
				zeta[i][t] = r.nextDouble() + small;
			}
		}
	}

	/*
	 * returns the value for the function lambda(zeta_y_t);
	 */
	public double lambdaZeta(int index, int t) {
		double value = 0.5/zeta[index][t] * (1/(1+Math.exp(-zeta[index][t])) - 0.5);
		try{
			MathUtils.check(value);
		} catch (Exception e) {
			System.err.println("zeta : " + zeta[index]);
		}
		return value;
	}
}
