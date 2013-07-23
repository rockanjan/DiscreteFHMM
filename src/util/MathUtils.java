package util;

import program.Main;

public class MathUtils {
	
	public static double dot(double[] a, double[] b) {
		double result = 0.0;
		if(a.length != b.length) {
			throw new RuntimeException("Dot product requires two vectors to have same length");
		}		
		for(int i=0; i<a.length; i++) {
			result += a[i] * b[i];			
		}
		return result;
	}
	
	public static double exp(double val) {
	    return Math.exp(val);
	    //return expApprox(val);
	}
	
	/*
	 * From the paper A Fast, Compact Approximation of the Exponential Function
	 * faster but not so accurate exp()
	 */
	private static double expApprox(double val) {
	    final long tmp = (long) (1512775 * val + (1072693248 - 60801));
	    return Double.longBitsToDouble(tmp << 32);
	}
	
	//log( exp(a) + exp(b) )
	public static double logsumexp(double a, double b) {
		if(Double.isInfinite(a)) {
			throw new RuntimeException("LogSum first term is infinite");
		}
		if(Double.isInfinite(b)) {
			throw new RuntimeException("LogSum second term is infinite");
		}		
		
		if( a > b) {
			return a + Math.log1p(Math.exp(b - a));
		} else {
			return b + Math.log1p(Math.exp(a - b));
		}
	}
	
	public static double logsumexp(double[] values) {
		double result = 0.0;
		double MAX = -Double.MAX_VALUE;
		for(int i=0; i<values.length; i++) {
			if(values[i] > MAX) {
				MAX = values[i];
			}
		}
		double expsum = 0.0;
		for(int i=0; i<values.length; i++) {
			expsum += Math.exp(values[i] - MAX);
		}
		result = MAX + Math.log(expsum); 
		return result;
	}
	
	public static double[][] expArray(double[][] array) {
		double[][] expArray = new double[array.length][array[0].length];
		for(int i=0; i<array.length; i++) {
			for(int j=0; j<array[0].length; j++) {
				expArray[i][j] = MathUtils.exp(array[i][j]);
			}
		}
		return expArray;
	}
	
	public static double expDot(double[] expWeights, double[] conditional) {
		double result = 1.0;
		for(int i=0; i<conditional.length; i++) {
			if(conditional[i] != 0) {
				result *= expWeights[i];
			}
		}
		if(Double.isInfinite(result)) {
			throw new RuntimeException("Error: expDot value is infinite");
		}
		return result;
	}
	
	public static double expDot(double[] expWeights, int state, int[] z) {
		double result = 1.0;
		result *= expWeights[0]; //bias
		result *= expWeights[1 + state]; //for the current hidden layer state
		int offset = Main.numStates;
		for(int i=1; i<=z.length; i++) {
			//0 in the first z layer has to be in the offset + 1
			result *= expWeights[ 1 + offset * i + z[i]];
		}
		if(Double.isInfinite(result)) {
			throw new RuntimeException("Error: expDot value is infinite");
		}
		return result;
	}
	
	/*
	 * adds the second matrix's entries to the first
	 */
	public static void addMatrix(double[][] target, double[][] source) {
		int m = target.length;
		int n = target[0].length;
		
		if(source.length != m || source[0].length != n) {
			throw new RuntimeException("Matrix dimensions mismatch");
		}
		for(int i=0; i<m; i++) {
			for(int j=0; j<n; j++) {
				target[i][j] += source[i][j];
			}
		}
	}
	
	public static double matrixDifferenceNorm(double[][] A, double[][] B) {
		int m = A.length;
		int n = A[0].length;
		if(B.length != m || B[0].length != n) {
			throw new RuntimeException("Matrix dimensions mismatch");
		}
		
		double differenceNorm = 0;
		for(int i=0; i<m; i++) {
			for(int j=0; j<n; j++) {
				double diff = A[i][j] - B[i][j];
				differenceNorm += diff * diff;
			}
		}
		return Math.sqrt(differenceNorm);
	}
	
	public static void matrixElementWiseMultiplication(double[][] A, double value) {
		for(int i=0; i<A.length; i++) {
			for(int j=0; j<A[0].length; j++) {
				A[i][j] *= value;
			}
		}
	}
	
	
	public static void main(String[] args) {
		double a = 0.12;
		double b = -20;
		System.out.println(a + Math.log(1 + exp(b - a)));
		System.out.println(logsumexp(a, b));
		double[] values = {a, b};
		System.out.println(logsumexp(values));
		System.out.println(Math.log(Math.exp(a) + Math.exp(b)));
		System.out.println(Math.log(expApprox(a) + expApprox(b)));
	}
}
