package util;

import java.util.List;

public class MyArray {

	public static void printTable(double[][] table) {
		System.out.println("Table...");
		for (int i = 0; i < table.length; i++) {
			for (int j = 0; j < table[i].length; j++) {
				System.out.print(table[i][j] + "\t");
			}
			System.out.println();
		}
	}
	
	public static void printVector(double[] vector, String title) {
		System.out.println("Vector: " + title);
		for(int i=0; i<vector.length; i++) {
			System.out.print(vector[i] + "\t");
		}
	}

	public static void printTable(double[][] table, String title) {
		System.out.println("Table : " + title);
		printTable(table);
	}

	public static void printExpTable(double[][] table, String title) {
		System.out.println("Table : " + title);
		printExpTable(table);
	}

	public static void printExpTable(double[][] table) {
		System.out.println("Table...");
		for (int i = 0; i < table.length; i++) {
			for (int j = 0; j < table[i].length; j++) {
				System.out.print(Math.exp(table[i][j]) + "\t");
			}
			System.out.println();
		}
	}

	public static double[] createVector(double[][] matrix) {
		// columnize: stack columns
		double[] vector = new double[matrix.length * matrix[0].length];
		int index = 0;
		for (int j = 0; j < matrix[0].length; j++) {
			for (int i = 0; i < matrix.length; i++) {
				vector[index++] = matrix[i][j];
			}
		}
		return vector;
	}

	public static double[][] createMatrix(double[] vector, int nrRows) {
		if (vector.length % nrRows != 0) {
			throw new RuntimeException(
					"Cannot create matrix for a vector of length "
							+ vector.length + " into " + nrRows + " rows.");
		}
		int nrColumns = (int) (vector.length / nrRows);
		double[][] matrix = new double[nrRows][nrColumns];
		int index = 0;
		for (int j = 0; j < nrColumns; j++) {
			for (int i = 0; i < nrRows; i++) {
				matrix[i][j] = vector[index++];
			}
		}
		return matrix;
	}

	public static double getL2NormSquared(double[] vector) {
		double norm = 0.0;
		for (int i = 0; i < vector.length; i++) {
			norm += Math.pow(vector[i], 2);
		}
		return norm;
	}
	
	public static double getSum(double[] vector) {
		double sum = 0.0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector[i];
		}
		return sum;
	}

	public static double[] getMinMaxOfMatrix(double[][] matrix) {
		double[] result = { Double.MAX_VALUE, -Double.MAX_VALUE };
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				if (matrix[i][j] > result[1]) {
					result[1] = matrix[i][j];
				}
				if (matrix[i][j] < result[0]) {
					result[0] = matrix[i][j];
				}
			}
		}
		return result;
	}

	public static double[] getMinMaxOfVector(double[] vector) {
		double[] result = { Double.MAX_VALUE, -Double.MAX_VALUE };
		for (int i = 0; i < vector.length; i++) {

			if (vector[i] > result[1]) {
				result[1] = vector[i];
			}
			if (vector[i] < result[0]) {
				result[0] = vector[i];
			}
		}
		return result;
	}
	
	public static double[][] getCloneOfMatrix(double[][] source) {
		double[][] clone = new double[source.length][source[0].length];
		for(int i=0; i<source.length; i++) {
			for(int j=0; j<source[0].length; j++) {
				clone[i][j] = source[i][j];
			}
		}
		return clone;
	}
	
	public static <T> void printTable(List<T> list) {
		for(int i=0; i<list.size(); i++) {
			System.out.print(list.get(i) + "\t");
		}
		System.out.println();
	}
	
	
}
