package model.inference;

public class VariationalParamZeta {
	int V; //vocab size
	double zeta[];
	public VariationalParamZeta(int V) {
		this.V = V;
		zeta = new double[V];
	}

	/*
	 * returns the value for the function lambda(zeta_k);
	 */
	public double lamdaZeta(int index) {
		return 0.5/zeta[index] * (1/(1+Math.exp(-zeta[index])) - 0.5);
	}
}
