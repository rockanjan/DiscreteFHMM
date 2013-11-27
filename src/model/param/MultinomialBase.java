package model.param;

import java.util.Random;

import javax.management.RuntimeErrorException;

import util.MathUtils;
import util.MyArray;
import util.Stats;

public abstract class MultinomialBase {
	//x,y == P(x given y)
	int x,y;
	public double[][] count;
	public double[][] oldCount; //for combining with adaptive weight for stochastic update
	
	public double[][] oldParams; //params before updating (for computing convergence by L1DiffNorm)
	public void initializeUniformCounts() {
		for(int i=0; i<x; i++) {
			for(int j=0; j<y; j++) {
				count[i][j] = 1.0;
			}
		}
	}
	
	public double[] getDistributionGivenState(int state) {
		double[] dist = new double[x];
		for(int i=0; i<x; i++) {
			dist[i] = Math.exp(count[i][state]);
		}
		return dist;
	}
	
	public MultinomialBase(int x, int y) {
		this.x = x; this.y = y;
		count = new double[x][y];
	}
	
	public double get(int x, int y) {
		return count[x][y];
	}
	
	public void set(int x, int y, double value) {
		count[x][y] = value;
	}
	
	public void addToCounts(int x, int y, double value) {
		count[x][y] += value;
	}
	
	public void cloneFrom(MultinomialBase source) {
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				count[j][i] = source.count[j][i];
			}
		}
	}
	
	//computes weighted average of sufficient statistics using older and newer counts
	//oldCount stores the old actual sufficient statistics (counts), 
	//count stores the recent probabilites (i.e parameters after normalizing)
	public void cloneWeightedFrom(MultinomialBase source, double weight) {
		oldParams = MyArray.getCloneOfMatrix(this.count);
		if(oldCount == null) {
			oldCount = MyArray.getCloneOfMatrix(source.count);
		}
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				//weighted expected counts
				count[j][i] = weight * source.count[j][i] + (1-weight) * oldCount[j][i];
			}
		} 
		//store the new expected counts for next iteration
		oldCount = MyArray.getCloneOfMatrix(count);		
	}
	
	public boolean equalsExact(MultinomialBase other) {
		boolean result = true;
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				if(count[j][i] != other.get(j,i)) {
					result = false;
				}
			}
		}
		return result;
	}
	
	public boolean equalsApprox(MultinomialBase other) {
		double precision = 1e-200;
		boolean result = true;
		for(int i=0; i<y; i++) {
			for(int j=0; j<x; j++) {
				if(Math.abs(count[j][i] - other.get(j,i)) > precision) {
					result = false;
				}
			}
		}
		return result;
	}
	
	public int getConditionedSize() {
		return x;
	}
	
	public int getConditionalSize() {
		return y;
	}
	
	public abstract void initializeRandom(Random r);
	public abstract void smooth();
	public abstract void normalize();
	public abstract void checkDistribution();
	public abstract void printDistribution();	
}
