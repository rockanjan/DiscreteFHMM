package corpus;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.management.RuntimeErrorException;

import program.Main;

import util.MyArray;

public class DiscreteSampler {
	Random random = new Random(Main.seed);
	List<Double> distribution;
	
	int[] alias; //stores the alias (other index whose mass is stacked at this position)
	double[] prob; //stores the remaining prob (can be between [0,1]) of the index after the alias method
	
	static int N;
	
	public DiscreteSampler(List<Double> distribution) {
		this.distribution = distribution;
		N = distribution.size();
		setupAliasMethod();
	}
	
	public DiscreteSampler(double[] dist) {
		this.distribution = new ArrayList<Double>();
		for(int i=0; i<dist.length; i++) {
			this.distribution.add(dist[i]);
		}		
		N = distribution.size();
		setupAliasMethod();
	}
	
	/*
	 * Alias method by Walker (1977)
	 * O(1) sampling time to sample a RV from the distribution
	 * algorithm at : http://www.keithschwarz.com/darts-dice-coins/
	 */
	private void setupAliasMethod() {
		List<Double> temp = new ArrayList<Double>();
		for(int i=0; i<N; i++) {
			temp.add(distribution.get(i) * N); //scale			
		}
		alias = new int[N];		
		Arrays.fill(alias, -1);
		prob = new double[N];
		Arrays.fill(prob, -1);
		
		for(int j=0; j<N-1; j++) {
			//MyArray.printTable(temp);
			int minIndex = -1;
			for(int k=0; k<N; k++) {
				if(temp.get(k) != -1 && temp.get(k) <=1) {
					minIndex = k;
					break;
				}				
			}
			
			int maxIndex = -1;
			for(int k=0; k<N; k++) {
				if(temp.get(k) != -1 && temp.get(k) >=1 && k != minIndex) {
					maxIndex = k;
					break;
				}				
			}
			prob[minIndex] = temp.get(minIndex);
			alias[minIndex] = maxIndex;
			if(maxIndex != -1) {
				temp.set(maxIndex, temp.get(maxIndex) - (1 - temp.get(minIndex)));
			}
			//disregard minIndex from consideration in next iter
			temp.set(minIndex, new Double(-1));
		}
		//there is a last probability whose prob[] is not set, set it to one
		for(int i=0; i<N; i++) {
			if(prob[i] == -1) {
				prob[i] = 1;
			}
		}
		//verify that for indices which do not have alias, prob must be 1
		for(int i=0; i<N; i++) {
			if(alias[i] == -1) {
				System.out.println("Alias = -1, prob= " + prob[i]);
				if(Math.abs(prob[i] - 1) > 1e-5) {
					throw new RuntimeException("Index with no alias does not have probability one, prob = " + prob[i]);
				}
			}
		}
	}
	
	//returns int between [0,N-1]
	public int sample() {
		int n = random.nextInt(N);
		double p = random.nextDouble();
		if(p > prob[n]) {
			return alias[n];
		} else {
			return n;
		}
	}
	
	public static void main(String[] args) {
		double[] counts = {20, 10, 5, 3, 2, 20, 50};
		int N = counts.length;
		double total = 0;
		for(int i=0; i<N; i++) {
			total += counts[i];
		}
		System.out.println("Original distribution");
		for(int i=0; i<N; i++) {
			counts[i] = counts[i]/total;
			System.out.format("%.4f \t", counts[i]);
		}
		System.out.println();
		
		DiscreteSampler sampler = new DiscreteSampler(counts);
		
		double[] sampledCount = new double[N];
		//sample
		int totalSamples = 100000;
		for(int i=0; i<totalSamples; i++) {
			sampledCount[sampler.sample()]++;
		}
		
		System.out.println("Sampled distribution");
		//renormalize
		for(int i=0; i<N; i++) {
			sampledCount[i] = sampledCount[i] / totalSamples;
			System.out.format("%.4f \t", sampledCount[i]);
		}
		System.out.println();		
	}
}
