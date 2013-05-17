package util;

public class MyArray {

	public static void printTable(double[][] table) {
		System.out.println("Table...");
		for(int i=0; i<table.length; i++) {
			for(int j=0; j<table[i].length; j++) {
				System.out.print(table[i][j] + "\t");
			}
			System.out.println();
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
		for(int i=0; i<table.length; i++) {
			for(int j=0; j<table[i].length; j++) {
				System.out.print(Math.exp(table[i][j]) + "\t");
			}
			System.out.println();
		}
	}
	
	public static double[] createVector(double[][] matrix) {
		//columnize: stack columns
		double[] vector = new double[matrix.length * matrix[0].length];
		int index = 0;
		for(int j=0; j<matrix[0].length; j++) {
			for(int i=0; i<matrix.length; i++) {
				vector[index++] = matrix[i][j];
			}
		}
		return vector;
	}
	
	public static double[][] createMatrix(double[] vector, int nrRows) {
		if(vector.length % nrRows != 0) {
			throw new RuntimeException("Cannot create matrix for a vector of length " + vector.length + " into " + nrRows + " rows.");
		}
		int nrColumns = (int) (vector.length / nrRows);
		double[][] matrix = new double[nrRows][nrColumns];
		int index = 0;
		for(int j=0; j<nrColumns; j++) {
			for(int i=0; i<nrRows; i++) {
				matrix[i][j] = vector[index++];
			}
		}
		return matrix;
	}
	
	public static double getL2NormSquared(double[] vector) {
		double norm = 0.0;
		for(int i=0; i<vector.length; i++) {
			norm += Math.pow(vector[i], 2);
		}
		return norm;
	}
}
