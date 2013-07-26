package model.inference;

import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import corpus.Instance;
import model.HMMBase;
import model.param.HMMParamBase;

public class VariationalParam {
	int iter = 0;
	public VariationalParamAlpha alpha;
	public VariationalParamZeta zeta;
	public HMMBase model;
	
	int M;
	int K;
	int V;
	
	public VariationalParam(HMMBase model) {
		this.model = model;
		M = model.nrLayers;
		K = model.nrStates;
		V = Corpus.corpusVocab.get(0).vocabSize;
		alpha = new VariationalParamAlpha();
		alpha.initializeRandom();
		zeta = new VariationalParamZeta(V);		
		zeta.initializeRandom();
	}
	
	public void optimizeInstanceParam(Instance instance) {
		//optimize shi's
		for(int m=0; m<M; m++) {
			for(int t=0; t<instance.T; t++) {
				for(int k=0; k<K; k++) {
					double sumThetaOverY = 0;
					for(int y=0; y<model.param.weights.vocabSize; y++) {
						sumThetaOverY += model.param.weights.get(m, k, y);
					}
					double updateValue = model.param.weights.get(m, k, instance.words[t][0]) - 0.5 * sumThetaOverY;
					for(int y=0; y<model.param.weights.vocabSize; y++) {
						double lambda = zeta.lamdaZeta(y);
						double sumY = 0;
						for(int n=0; n<M; n++) {
							if(n==m) {
								continue;
							}
							double dotProd = MathUtils.dot(model.param.weights.getStateVector(n, y), instance.posteriors[m][t]);
							MathUtils.check(dotProd);
							sumY += model.param.weights.getStateVector(m, y)[k] * dotProd;
							MathUtils.check(sumY);
						}
						double[] thetaMY = model.param.weights.getStateVector(m, y);
						double delta = MathUtils.diag(MathUtils.getOuterProduct(thetaMY, thetaMY))[k];
						sumY += delta;
						
						double sumOverM = 0;
						for(int n=0; n<M; n++) {
							sumOverM += model.param.weights.get(m, k, y);
							MathUtils.check(sumOverM);
						}
						sumY -= 2 * alpha.alpha * sumOverM;
						MathUtils.check(sumY);
						updateValue -= lambda * sumY;
						MathUtils.check(updateValue);
					}
					instance.varParamObs.shi[m][t][k] = updateValue;
				}
			}
			//TODO: decide if computing posteriors in bulk is better or after each layer
			//re-estimate state posteriors for this layer
			instance.forwardBackwardList.get(m).doInference();				
		}
	}
	
	public void optimizeCorpusParam() {
		int maxIter = 1;
		for(int iter=0; iter<maxIter; iter++) {
			//optimize zetas
			for(int y=0; y<model.param.weights.vocabSize; y++) {
				zeta.zeta[y] = 0;
				int totalT = 0;
				for(int i=0; i<Corpus.trainInstanceEStepSampleList.size(); i++) {
					Instance inst = Corpus.trainInstanceEStepSampleList.get(i);
					//WARNING: do not use instance from above here. 
					//TODO: refactor the code
					for(int t=0; t<inst.T; t++) {
						totalT++;
						zeta.zeta[y] += Math.pow(alpha.alpha, 2);
						for(int m=0; m<M; m++) {
							for(int n=0; n<M; n++) {
								if(m==n) continue;
								double dotThetaMStateM = MathUtils.dot(model.param.weights.getStateVector(m, y),
										inst.forwardBackwardList.get(m).posterior[t]);
								double dotThetaNStateN =  MathUtils.dot(model.param.weights.getStateVector(n, y),
										inst.forwardBackwardList.get(n).posterior[t]);
								zeta.zeta[y] += dotThetaMStateM * dotThetaNStateN;
							}
							
							double thetaMDiagStateMthetaM = MathUtils.vectorTransposeMatrixVector(
									model.param.weights.getStateVector(m, y), 
									MathUtils.diag(inst.forwardBackwardList.get(m).posterior[t]), 
									model.param.weights.getStateVector(m, y));
							zeta.zeta[y] += thetaMDiagStateMthetaM;
							
							double dotStateMThetaM = MathUtils.dot(
									inst.forwardBackwardList.get(m).posterior[t],
									model.param.weights.getStateVector(m, y)
									);
							
							zeta.zeta[y] -= 2 * alpha.alpha * dotStateMThetaM;
						}
					}
				}
				//value should not be negative because we still have to take squareroot
				if(zeta.zeta[y] < 0) {
					System.err.println("zeta value negative found : " + zeta.zeta[y]);
				}
				zeta.zeta[y] = Math.sqrt(zeta.zeta[y]);
			}
		
			//optimize alpha
			double sumTMY = 0;
			for(int i=0; i<Corpus.trainInstanceEStepSampleList.size(); i++) {
				Instance inst = Corpus.trainInstanceEStepSampleList.get(i);
				for(int t=0; t<inst.T; t++) {
					for(int y=0; y<model.param.weights.vocabSize; y++) {
						for(int m=0; m<M; m++) {
							double dotStateMThetaM = MathUtils.dot(
									inst.forwardBackwardList.get(m).posterior[t],
									model.param.weights.getStateVector(m, y)
									);
							sumTMY += zeta.lamdaZeta(y) * dotStateMThetaM; 
						}
					}
				}
			}
			double numerator = 0;
			numerator = Corpus.trainInstanceEStepSampleList.numberOfTokens * (0.5 * model.param.weights.vocabSize - 1)  
					+ 2 * sumTMY;
			double denominator = 0;
			for(int y=0; y<model.param.weights.vocabSize; y++) {
				denominator += zeta.lamdaZeta(y);
			}
			denominator = 2 * Corpus.trainInstanceEStepSampleList.numberOfTokens * denominator;		
			alpha.alpha = numerator / denominator;
		}
	}
}
