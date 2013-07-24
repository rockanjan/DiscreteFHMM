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
		alpha.initializeRandom();
		zeta = new VariationalParamZeta(V);				
	}
	
	public void optimize() {
		int maxIter = 10;
		for(int iter=0; iter<maxIter; iter++) {
			//optimize shi's
			for(int m=0; m<M; m++) {
				for(int t=0; t<T; t++) {
					for(int k=0; k<K; k++) {
						double sumThetaOverY = 0;
						for(int y=0; y<hmmParam.weights.vocabSize; y++) {
							sumThetaOverY += hmmParam.weights.get(m, k, y);
						}
						obs.shi[m][t][k] = hmmParam.weights.get(m, k, instance.words[t][0]) - 0.5 * sumThetaOverY;
						for(int y=0; y<hmmParam.weights.vocabSize; y++) {
							double lambda = zeta.lamdaZeta(y);
							double sumY = 0;
							for(int n=0; n<M; n++) {
								if(n==m) {
									continue;
								}
								double dotProd = MathUtils.dot(hmmParam.weights.getStateVector(n, y), instance.posteriors[m][t]);
								sumY += hmmParam.weights.getStateVector(m, y)[k] * dotProd;
							}
							double[] thetaMY = hmmParam.weights.getStateVector(m, y);
							double delta = MathUtils.diag(MathUtils.getOuterProduct(thetaMY, thetaMY))[k];
							sumY += delta;
							
							double sumOverM = 0;
							for(int n=0; n<M; n++) {
								sumOverM += hmmParam.weights.get(m, k, y); 
							}
							sumY -= 2 * alpha.alpha * sumOverM;
							obs.shi[m][t][k] -= lambda * sumY;
						}
					}
				}
				//TODO: decide if computing posteriors in bulk better or after each layer
				//re-estimate state posteriors for this layer
				instance.forwardBackwardList.get(m).doInference();				
			}
			
			//optimize zetas
			//force the expression to be zero for each timestep
			for(int i=0; i<Corpus.trainInstanceEStepSampleList.size(); i++) {
				Instance inst = Corpus.trainInstanceEStepSampleList.get(i);
				//WARNING: do not use instance from above here. 
				//TODO: refactor the code
				for(int t=0; t<inst.T; t++) {
					for(int y=0; y<hmmParam.weights.vocabSize; y++) {
						zeta.zeta[y] = Math.pow(alpha.alpha, 2);
						for(int m=0; m<M; m++) {
							for(int n=0; n<M; n++) {
								if(m==n) continue;
								double dotThetaMStateM = MathUtils.dot(hmmParam.weights.getStateVector(m, y),
										inst.forwardBackwardList.get(m).posterior[t]);
								double dotThetaNStateN =  MathUtils.dot(hmmParam.weights.getStateVector(n, y),
										inst.forwardBackwardList.get(n).posterior[t]);
								zeta.zeta[y] += dotThetaMStateM * dotThetaNStateN;
							}
							
							double thetaMDiagStateMthetaM = MathUtils.vectorTransposeMatrixVector(
									hmmParam.weights.getStateVector(m, y), 
									MathUtils.diag(inst.forwardBackwardList.get(m).posterior[t]), 
									hmmParam.weights.getStateVector(m, y));
							zeta.zeta[y] += thetaMDiagStateMthetaM;
							
							double dotStateMThetaM = MathUtils.dot(
									inst.forwardBackwardList.get(m).posterior[t],
									hmmParam.weights.getStateVector(m, y)
									);
							
							zeta.zeta[y] -= 2 * alpha.alpha * dotStateMThetaM;
						}
					}
				}
			}
		
		
			//optimize alpha
			double sumTMY = 0;
			for(int i=0; i<Corpus.trainInstanceEStepSampleList.size(); i++) {
				Instance inst = Corpus.trainInstanceEStepSampleList.get(i);
				for(int t=0; t<inst.T; t++) {
					for(int y=0; y<hmmParam.weights.vocabSize; y++) {
						for(int m=0; m<M; m++) {
							double dotStateMThetaM = MathUtils.dot(
									inst.forwardBackwardList.get(m).posterior[t],
									hmmParam.weights.getStateVector(m, y)
									);
							sumTMY += zeta.lamdaZeta(y) * dotStateMThetaM; 
						}
					}
				}
			}
			double numerator = 0;
			numerator = Corpus.trainInstanceEStepSampleList.numberOfTokens * (0.5 * hmmParam.weights.vocabSize - 1)  
					+ 2 * sumTMY;
			double denominator = 0;
			for(int y=0; y<hmmParam.weights.vocabSize; y++) {
				denominator += zeta.lamdaZeta(y);
			}
			denominator = 2 * Corpus.trainInstanceEStepSampleList.numberOfTokens * denominator;		
			alpha.alpha = numerator / denominator;
		}
	}
}
