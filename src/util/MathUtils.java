package util;

import config.Config;
import program.Main;

public class MathUtils {
	
	public static double dot(double[] a, double[] b) {
		double result = 0.0;
		if(a.length != b.length) {
			throw new RuntimeException(String.format("Dot product requires two vectors to have same length, found %d vs %d", a.length, b.length));
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
		int offset = Config.numStates;
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
	

	public static double[][] weightedAverageMatrix(double[][] first, double[][] second, double weight) {
		if(weight <=0 || weight > 1) {
			throw new RuntimeException(String.format("Invalid weight %f, should be (0,1]", weight));
		}
		int m = first.length;
		int n = first[0].length;
		
		if(second.length != m || second[0].length != n) {
			throw new RuntimeException("Matrix dimensions mismatch");
		}
		double[][] averageMatrix = new double[m][n];
		for(int i=0; i<m; i++) {
			for(int j=0; j<n; j++) {
				averageMatrix[i][j] = (1-weight) * first[i][j] + weight * second[i][j];
			}
		}
		return averageMatrix;
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
	
	public static double[][] getOuterProduct(double[] v1, double v2[]) {
		double result[][] = new double[v1.length][v2.length];
		for(int i=0; i<v1.length; i++) {
			for(int j=0; j<v2.length; j++) {
				result[i][j] = v1[i] * v2[j];
			}
		}
		return result;
	}
	/*
	 * returns result for x'Ay
	 */
	public static double vectorTransposeMatrixVector(double[] v1, double[][] matrix, double[] v2) {
		double result = 0;
		if(v1.length != matrix.length) {
			throw new RuntimeException(String.format("vector1 length: %d, matrix row size: %d", v1.length, matrix.length));
		}
		if(v2.length != matrix[0].length) {
			throw new RuntimeException(String.format("vector2 length: %d, matrix column size: %d", v2.length, matrix[0].length));
		}
		for(int i=0; i<matrix.length; i++) {
			for(int j=0; j<matrix[0].length; j++) {
				result += v1[i] * matrix[i][j] * v2[j]; 
				check(result);
			}
		}
		return result;
	}
	
	public static double trace(double[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		if(m != n) {
			throw new RuntimeException(String.format("Trying to find trace of a rectangular matrix: m=%d, n=%d", m, n));
		}
		double sum = 0;
		for(int i=0; i<m; i++) {
			sum += matrix[i][i];
		}
		return sum;
	}
	
	public static double[] getVectorFromMatrixDiagonalEntries(double[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		if(m != n) {
			throw new RuntimeException(String.format("Trying to get vector from diagonal entries of a rectangular matrix: m=%d, n=%d", m, n));
		}
		double[] vector = new double[m];
		for(int i=0; i<m; i++) {
			vector[i] = matrix[i][i];
		}
		return vector;
	}
	
	public static double[] addVectors(double[] v1, double[] v2) {
		int m=v1.length;
		int n=v2.length;
		if(n != m) {
			throw new RuntimeException(String.format("Error in adding vectors, dimension mismatch %d vs %d", m, n));
		}
		
		double result[] = new double[m];
		for(int i=0; i<m; i++) {
			result[i] = v1[i] + v2[i];
		}
		return result;
	}
	
	/*
	 * gives a vector formed by the diagonal elements as entries
	 */
	public static double[] diag(double[][] matrix) {
		return getVectorFromMatrixDiagonalEntries(matrix);
	}
	
	/*
	 * gives a matrix formed by entries of vector in the diagonals of the matrix, will off-diagonal entries zero
	 */
	public static double[][] diag(double[] vector) {
		double[][] matrix = new double[vector.length][vector.length];
		for(int i=0; i<vector.length; i++) {
			matrix[i][i] = vector[i];
		}
		return matrix;
	}
	
	public static void check(double value) {
		if(Double.isInfinite(value)) {
			throw new RuntimeException("Infinite value found");
		}
		if(Double.isNaN(value)) {
			throw new RuntimeException("NaN value found");
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
