package util;

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
}
