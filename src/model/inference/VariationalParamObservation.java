package model.inference;

import java.util.Random;

import corpus.Corpus;

import program.Main;

public class VariationalParamObservation {
	int M,T,K;
	public double shi[][][];
	
	/*
	 * @param M number of layers
	 * @param T timesteps
	 * @param K number of states
	 */
	public VariationalParamObservation(int M, int T, int K) {
		this.M = M;
		this.T = T;
		this.K = K;
		shi = new double[M][T][K];
	}
	
	public void initializeRandom() {
		Random r = new Random(Main.seed);
		double small = 1e-100;
		for(int m=0; m<M; m++) {
			for(int t=0; t<T; t++) {
				double sum = 0;
				for(int k=0; k<K; k++) {
					shi[m][t][k] = r.nextDouble() + small;
					sum += shi[m][t][k];
				}
				//normalize
				for(int k=0; k<K; k++) {
					shi[m][t][k] = Math.log(shi[m][t][k]/sum);
				}
			}
		}
	}
	
	public void initializeUniform() {
		for(int m=0; m<M; m++) {
			for(int t=0; t<T; t++) {
				for(int k=0; k<K; k++) {
					shi[m][t][k] = 1/Corpus.corpusVocab.get(0).vocabSize;					
				}
			}
		}
	}
}
