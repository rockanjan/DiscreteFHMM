package model.inference;

import model.param.HMMParamBase;
import config.Config;
import corpus.Instance;
import corpus.WordClass;

public class VariationalParamObservation {
	int T;
	int[] states;
	public double shi[][][];
	
	public VariationalParamObservation(int[] states, int T) {
		this.states = states;
		this.T = T;
		shi = new double[states.length][T][];
		for(int m=0; m<states.length; m++) {
			for(int t=0; t<T; t++) {
				shi[m][t] = new double[states[m]];
			}
		}
	}
	
	public void initializeFromObsAndClassParam(HMMParamBase param, Instance instance) {
		for(int m=0; m<states.length; m++) {
			for(int t=0; t<T; t++) {
				double sum = 0;
				for(int k=0; k<states[m]; k++) {
					int clusterId = WordClass.wordIndexToClusterIndex.get(instance.words[t][0]);
					shi[m][t][k] = param.weights.get(m, k, instance.words[t][0]) + param.weightsClass.get(m, k, clusterId);					
					sum += Math.exp(shi[m][t][k]); //cached exponentiated result
				}
				//normalize
				for(int k=0; k<states[m]; k++) {
					shi[m][t][k] = shi[m][t][k] - Math.log(sum);
				}
			}
		}
	}
	
	public void initializeRandom() {
		double small = 1e-100;
		for(int m=0; m<states.length; m++) {
			for(int t=0; t<T; t++) {
				double sum = 0;
				for(int k=0; k<states[m]; k++) {
					shi[m][t][k] = Config.random.nextDouble() + small;
					sum += shi[m][t][k];
				}
				//normalize
				for(int k=0; k<states[m]; k++) {
					shi[m][t][k] = Math.log(shi[m][t][k]/sum);
				}
			}
		}
	}
	
	public void initializeUniform() {
		for(int m=0; m<states.length; m++) {
			for(int t=0; t<T; t++) {
				for(int k=0; k<states[m]; k++) {
					shi[m][t][k] = 1/states[m];
				}
			}
		}
	}
}
