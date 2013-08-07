package model.inference;

import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;
import model.HMMBase;
import model.param.HMMParamBase;

public class VariationalParam {
	public VariationalParamObservation varParamObs;
	public VariationalParamAlpha alpha;
	public HMMBase model;
	public Instance instance;
	
	int M;
	int K;
	int V;
	int T;
	
	public VariationalParam(HMMBase model, Instance instance) {
		this.model = model;
		M = model.nrLayers;
		K = model.nrStates;
		V = Corpus.corpusVocab.get(0).vocabSize;
		this.T = instance.T;
		this.instance = instance;
		
		varParamObs = new VariationalParamObservation(M, T, K);
		varParamObs.initializeRandom();
		alpha = new VariationalParamAlpha(T);
		alpha.initializeRandom();		
	}
	
	public void optimize() {
		optimizeAlpha();
		optimizeParamObs();
	}
	
	public void optimizeParamObs() {
		double minExtreme = Double.MAX_VALUE;
		double maxExtreme = -Double.MAX_VALUE;
		//optimize shi's
		double shiL1NormInstance = 0;
		for(int m=0; m<M; m++) {
			for(int t=0; t<instance.T; t++) {
				double[] sumOverNYt = new double[K];
				for(int k=0; k<K; k++) {
					for(int n=0; n<M; n++) {				
						sumOverNYt[k] += model.param.weights.get(n, k, instance.words[t][0]);
					}
				}
				
				double[] sumOverNnotM = new double[K];
				for(int k=0; k<K; k++) {
					for(int n=0; n<M; n++) {
						if(n != m) {
							sumOverNnotM[k] += varParamObs.shi[n][t][k];
						}
					}
				}
				
				//TODO: might underflow
				double[] sumOverY = new double[K];
				for(int y=0; y<V; y++) {
					double prod = 1.0;
					for(int n=0; n<M; n++) {
						if(n != m) {
							prod *= MathUtils.dot(model.param.expWeights.getStateVector(n, y), 
									instance.forwardBackwardList.get(n).posterior[t]);
							if(prod == 0) {
								throw new RuntimeException("Underflow");
							}
						}
					}
					for(int k=0; k<K; k++) {
						sumOverY[k] += prod * model.param.expWeights.getStateVector(m, y)[k];
					}
				}
				double normalizer = 0;
				double maxOverK = -Double.MAX_VALUE;
				double[] updateValue = new double[K];
				for(int k=0; k<K; k++) {
					updateValue[k] = sumOverNYt[k] - sumOverNnotM[k] - alpha.alpha[t] * sumOverY[k];
					/*
					System.out.println(String.format("updateValue=%.2f, sumNYt=%.2f, sumNot=%.2f, alpha, %.2f, sumY=%.2f", updateValue[k], 
							sumOverNYt[k], sumOverNnotM[k], alpha.alpha[t], sumOverY[k]));
					*/
					if(updateValue[k] > maxOverK) {
						maxOverK = updateValue[k];
					}
				}
				normalizer = MathUtils.logsumexp(updateValue);
				//System.out.println("Normalizer : " + normalizer);
				//System.out.println("MaxoverK " + maxOverK);
				//normalize and update				
				for(int k=0; k<K; k++) {
					double oldValue = varParamObs.shi[m][t][k];
					//varParamObs.shi[m][t][k] = updateValue[k] - maxOverK;
					varParamObs.shi[m][t][k] = updateValue[k] - normalizer;
					MathUtils.check(varParamObs.shi[m][t][k]);
					shiL1NormInstance += Math.abs(oldValue - varParamObs.shi[m][t][k]);					
				}
			}							
		}	
		InstanceList.shiL1NormAll += shiL1NormInstance;		
	}
	
	public void optimizeAlpha() {
		double alphaL1Norm = 0;
		for(int t=0; t<T; t++) {
			double sumY = 0;
			for(int y=0; y<V; y++) {
				double prodM = 1.0;
				for(int m=0; m<M; m++) {
					double dotStateMExpThetaM = MathUtils.dot(
							instance.forwardBackwardList.get(m).posterior[t],
							model.param.expWeights.getStateVector(m, y)
							);
					prodM = prodM * dotStateMExpThetaM;
					if(prodM == 0) {
						throw new RuntimeException("Underflow");
					}
				}
				sumY += prodM;
			}	
			double oldAlpha = alpha.alpha[t];
			if(sumY <= 0) {
				System.err.println(String.format("sumY = %f, setting small value", sumY));
				sumY = 1e-200;
			}
			alpha.alpha[t] = 1/sumY;
			if(alpha.alpha[t] == 0) {
				System.err.println("alpha is zero");
				alpha.alpha[t] = 1e-200;
			}
			alphaL1Norm += Math.abs(alpha.alpha[t] - oldAlpha);
			MathUtils.check(alpha.alpha[t]);
		}
		InstanceList.alphaL1NormAll += alphaL1Norm;
	}
}
