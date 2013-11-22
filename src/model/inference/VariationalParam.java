package model.inference;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import model.HMMBase;
import util.MathUtils;
import corpus.Corpus;
import corpus.Instance;
import corpus.WordClass;

public class VariationalParam {
	public VariationalParamObservation varParamObs;
	public VariationalParamAlpha alphaY; //words
	public VariationalParamAlpha alphaC; //classes
	public HMMBase model;
	public Instance instance;
	
	int M;
	int K;
	int V;
	int T;
	int C;
	
	//for a fixed t, cache for each Y the dot product (S_tn, exp(theta_nY)) for all layers n 
	Map<Integer, Double> prodCache; //wordIndex to value
	Map<Integer, Double> prodCacheClass;
	
	public VariationalParam(HMMBase model, Instance instance) {
		this.model = model;
		M = model.nrLayers;
		K = model.nrStates;
		V = Corpus.corpusVocab.get(0).vocabSize;
		this.C = model.nrClasses;
		this.T = instance.T;
		this.instance = instance;
		
		varParamObs = new VariationalParamObservation(M, T, K);
		//varParamObs.initializeRandom();
		varParamObs.initializeFromObsParam(model.param, instance);
		alphaY = new VariationalParamAlpha(T); //actually phi in derivation
		alphaC = new VariationalParamAlpha(T); //actually phi in derivation
		
		alphaY.initializeUniform(1.0); 
		alphaC.initializeUniform(1.0);
	}
	
	public void optimize() {
		for(int t=0; t<instance.T; t++) {
			if(M > 1) {
				createCacheLogFix(t); //for Y
				createCacheLogFixClass(t); //for class
				optimizeAlphaY(t);
				optimizeAlphaC(t);
			}
			//optimizeParamObs(t);
			optimizeParamObsNew(t);
			clearCache();
		}
	}
	
	public void createCacheLogFix(int t) {
		prodCache = new HashMap<Integer, Double>();
		int wordCluster = WordClass.wordIndexToClusterIndex.get(instance.words[t][0]);
		Set<Integer> wordsInCluster = WordClass.clusterIndexToWordIndices.get(wordCluster);
		for(int y : wordsInCluster) {
			double prodLog = 0.0;
			for(int n=0; n<M; n++) {
				double dot = MathUtils.dot(model.param.expWeights.getStateVector(n, y), 
						instance.forwardBackwardList.get(n).posterior[t]);
				if(dot == 0) {
					dot = Math.log(1e-200);
				} else {
					dot = Math.log(dot);
				}
				prodLog = prodLog + dot;
			}
			prodCache.put(y, Math.exp(prodLog));
		}
	}
	
	public void createCacheLogFixClass(int t) {
		prodCacheClass = new HashMap<Integer, Double>();
		for(int c=0; c<C; c++) {
			double prodLog = 0.0;
			for(int n=0; n<M; n++) {
				double dot = MathUtils.dot(model.param.expWeightsClass.getStateVector(n, c), 
						instance.forwardBackwardList.get(n).posterior[t]);
				if(dot == 0) {
					dot = Math.log(1e-200);
				} else {
					dot = Math.log(dot);
				}
				prodLog = prodLog + dot;
			}
			prodCacheClass.put(c, Math.exp(prodLog));
		}
	}
	
	public void clearCache() {
		prodCache = null;
		prodCacheClass = null;
	}
	
	
	public void optimizeParamObsNew(int t) {
		if(M > 1) {
			// shi_mt = theta_mt + theta _mCt 
			// - phi_tY sum_Y_in_Ct( (prod_n!=m ( <s_tn>dot exp(theta_nY) * exp(theta_mY))))
			// - phi_tC sum_C(prod_n!=m ( <s_tn> dot exp(theta_nC) * exp(theta_mC)))
			
			for(int m=0; m<M; m++) {
				
				//for words in the cluster of current word
				double[] sumOverY = new double[K];
				int wordCluster = WordClass.wordIndexToClusterIndex.get(instance.words[t][0]);
				Set<Integer> wordsInCluster = WordClass.clusterIndexToWordIndices.get(wordCluster);
				for(int y : wordsInCluster) {
					double allProd = prodCache.get(y);
					double prod = allProd / MathUtils.dot(model.param.expWeights.getStateVector(m, y), 
							instance.forwardBackwardList.get(m).posterior[t]); //all prod except m'th layer
					for(int k=0; k<K; k++) {
						sumOverY[k] += prod * model.param.expWeights.getStateVector(m, y)[k];
					}
				}
				
				//for class
				double[] sumOverC = new double[K];
				for(int c=0; c<C; c++) {
					double allProd = prodCacheClass.get(c);
					double prod = allProd / MathUtils.dot(model.param.expWeightsClass.getStateVector(m, c), 
							instance.forwardBackwardList.get(m).posterior[t]); //all prod except m'th layer
					for(int k=0; k<K; k++) {
						sumOverY[k] += prod * model.param.expWeights.getStateVector(m, c)[k];
					}
				}
				
				
				double normalizer = 0;
				double maxOverK = -Double.MAX_VALUE;
				double[] updateValue = new double[K];
				for(int k=0; k<K; k++) {
					double prodY = alphaY.alpha[t] * sumOverY[k];
					double prodC = alphaC.alpha[t] * sumOverC[k];
					
					int currentWordIndex = instance.words[t][0];
					int currentClusterIndex = WordClass.wordIndexToClusterIndex.get(currentWordIndex);
					updateValue[k] = model.param.weights.get(m, k, currentWordIndex) 
							+ model.param.weightsClass.get(m, k, currentClusterIndex) - prodY - prodC;
					if(updateValue[k] > maxOverK) {
						maxOverK = updateValue[k];
					}
				}
				normalizer = MathUtils.logsumexp(updateValue);
				//normalize and update				
				for(int k=0; k<K; k++) {
					//varParamObs.shi[m][t][k] = updateValue[k] - maxOverK;
					varParamObs.shi[m][t][k] = updateValue[k] - normalizer;
					MathUtils.check(varParamObs.shi[m][t][k]);
				}
			}
		} else {
			for(int m=0; m<M; m++) {
				double normalizer = 0;
				double[] updateValue = new double[K];
				for(int k=0; k<K; k++) {
					updateValue[k] = model.param.weights.get(m, k, instance.words[t][0]);
				}
			}
		}
	}
	
	public void optimizeAlphaY(int t) {
		double sumY = 0;
		for(Integer y : prodCache.keySet()) {
			double prodM = prodCache.get(y);
			sumY += prodM;
		}	
		if(sumY <= 0) {
			System.err.println(String.format("WARNING: sumY = %f, setting small value", sumY));
			sumY = 1e-300;
		}
		alphaY.alpha[t] = 1/sumY;
		if(alphaY.alpha[t] == 0) {
			System.err.println("WARNING: alpha is zero");
			alphaY.alpha[t] = 1e-300;
		}
		MathUtils.check(alphaY.alpha[t]);		
	}
	
	public void optimizeAlphaC(int t) {
		double sumC = 0;
		for(Integer c : prodCacheClass.keySet()) {
			double prodM = prodCacheClass.get(c);
			sumC += prodM;
		}	
		if(sumC <= 0) {
			System.err.println(String.format("WARNING: sumC = %f, setting small value", sumC));
			sumC = 1e-300;
		}
		alphaC.alpha[t] = 1/sumC;
		if(alphaC.alpha[t] == 0) {
			System.err.println("WARNING: alpha is zero");
			alphaC.alpha[t] = 1e-300;
		}
		MathUtils.check(alphaC.alpha[t]);		
	}
}
