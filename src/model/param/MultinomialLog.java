package model.param;

import java.util.Random;

import util.MathUtils;
import util.MyArray;
import util.Stats;
import config.Config;

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
		double small = 1e-10;
		for(int i=0; i<y; i++) {
			double sum = 0;
			for(int j=0; j<x; j++) {
				count[j][i] = r.nextDouble() + small;
				//count[j][i] = 1.0;
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
		//add one smoothing
		double small = 0.1 / (Config.numStates * Config.numStates);
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				count[j][i] += small;				
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
				MathUtils.check(sum);
			}
			for(int j=0; j<x; j++) {
				count[j][i] = count[j][i] / sum;
				double actualCount = count[j][i];
				MathUtils.check(count[j][i]);
				if(count[j][i] == 0) {
					Stats.totalFixes++;
					count[j][i] = Math.log(1e-50);
				} else {
					count[j][i] = Math.log(count[j][i]);
				}
				
				try {
				MathUtils.check(count[j][i]);
				} catch(Exception e) {
					System.out.println("actual = " + actualCount + " " + count[j][i]);
					System.exit(-1);
				}
			}
		}
		//MyArray.printTable(count);
		checkDistribution();
	}
	
	@Override
	public void checkDistribution() {
		double tolerance = 1e-2;
		
		for(int i=0; i<y; i++) {
			double sum = 0;
			for(int j=0; j<x; j++) {
				sum += Math.exp(count[j][i]);
				try{
					MathUtils.check(sum);
				} catch(Exception e) {
					System.err.println(count[j][i]);
				}
			}
			
			if(Double.isNaN(sum)) {
				throw new RuntimeException("Distribution sums to NaN");
			}
			if(Double.isInfinite(sum)) {
				throw new RuntimeException("Distribution sums to Infinite");
			}
			if(Math.abs(sum - 1.0) > tolerance) {
				//System.err.println("Distribution sums to : " + sum);
				MyArray.printExpTable(count);
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
