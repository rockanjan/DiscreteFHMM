package model.inference;

import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import corpus.Instance;
import model.param.HMMParamBase;

public class VariationalParam {
	public VariationalParamObservation obs;
	public VariationalParamAlpha alpha;
	public VariationalParamZeta zeta;
	public Instance instance;
	HMMParamBase hmmParam;
	
	int M;
	int T;
	int K;
	int V;
	
	public VariationalParam(Instance instance) {
		this.instance = instance;
		this.hmmParam = instance.model.param;
		M = hmmParam.nrLayers;
		T = instance.T;
		K = instance.model.nrStates;
		V = Corpus.corpusVocab.get(0).vocabSize;
		
		obs = new VariationalParamObservation(M, T, K);
		obs.initializeRandom();
		alpha = new VariationalParamAlpha();
		zeta = new VariationalParamZeta(V);				
	}
	
	public void optimize() {
		int maxIter = 10;
		for(int iter=0; iter<maxIter; iter++) {
			//optimize shi's
			for(int m=0; m<M; m++) {
				for(int t=0; t<T; t++) {
					for(int k=0; k<K; k++) {
						double sumOverY = 0;
						for(int y=0; y<hmmParam.weights.vocabSize; y++) {
							sumOverY += hmmParam.weights.get(m, k, y);
						}
						obs.shi[m][t][k] = hmmParam.weights.get(m, k, instance.words[t][0]) - 0.5 * sumOverY;  
					}
				}
			}
		}
	}
}
