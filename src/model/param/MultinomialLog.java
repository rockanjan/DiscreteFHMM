package model.param;

import java.util.Random;
import util.MyArray;
import util.Stats;

/*
 * Stores all probabilities in log
 */
public class MultinomialLog extends MultinomialBase{
	public MultinomialLog(int x, int y) {
		super(x, y);	
	}
	
	@Override
	public void initializeRandom(Random r) {
		System.out.println("initializing multinomial log");
		double small = 1e-100;
		for(int i=0; i<y; i++) {
			double sum = 0;
			for(int j=0; j<x; j++) {
				count[j][i] = r.nextDouble() + small;
				sum += count[j][i];
			}
			//normalize
			for(int j=0; j<x; j++) {
				count[j][i] = Math.log(count[j][i]) - Math.log(sum);
			}
		}
		checkDistribution();
	}
	
	@Override
	public void smooth() {
		//hyperparameter
		double small = 100;
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				if(count[j][i] == 0) {
					Stats.totalFixes++;
					count[j][i] = small;
				}
			}
		}		
	}
	
	//when reached here, the counts are normal (no logs)
	@Override
	public void normalize() {
		smooth();
		for(int i=0; i<y; i++) {
			double sum = 0;
			for(int j=0; j<x; j++) {
				sum += count[j][i];
			}
			//MyArray.printTable(count);
			//normalize
			if(sum == 0) {
				throw new RuntimeException("Sum = 0 in normalization");
			}
			if(Double.isInfinite(sum)) {
				throw new RuntimeException("Sum is infinite in normalization");
			}
			if(Double.isNaN(sum)) {
				throw new RuntimeException("Sum is NaN in normalization");
			}
			for(int j=0; j<x; j++) {
				count[j][i] = count[j][i] / sum;
				if(Double.isNaN(count[j][i])) {
					System.err.format("count[%d][%d] = %f\n", j,i,count[j][i]);
					System.err.format("sum = %f\n", sum);
					throw new RuntimeException("Probability after normalization is NaN");
				}
				if(count[j][i] == 0) {
					//System.err.println("Prob distribution zero after normalization");
					Stats.totalFixes++;
					//fix
					count[j][i] = -Double.MAX_EXPONENT;
				} else {
					count[j][i] = Math.log(count[j][i]);
				}
			}
		}
		//MyArray.printTable(count);
		checkDistribution();
	}
	
	@Override
	public void checkDistribution() {
		double tolerance = 1e-5;
		
		for(int i=0; i<y; i++) {
			double sum = 0;
			for(int j=0; j<x; j++) {
				sum += Math.exp(count[j][i]);
			}
			
			if(Double.isNaN(sum)) {
				throw new RuntimeException("Distribution sums to NaN");
			}
			if(Double.isInfinite(sum)) {
				throw new RuntimeException("Distribution sums to Infinite");
			}
			if(Math.abs(sum - 1.0) > tolerance) {
				//System.err.println("Distribution sums to : " + sum);
				throw new RuntimeException("Distribution sums to : " + sum);
			}
		}
	}
	
	@Override
	public void printDistribution() {
		MyArray.printExpTable(count);
	}
	
	
	public static void main(String[] args) {
		MultinomialLog ml = new MultinomialLog(10, 20);
		ml.initializeRandom(new Random());
		ml.checkDistribution();
	}	
}
