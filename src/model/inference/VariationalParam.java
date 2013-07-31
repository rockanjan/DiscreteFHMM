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
	
	public static double shiL1NormAll=0;
	
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
		double shiL1NormInstance = 0;
		zeta.clearLambdaCache();
		for(int m=0; m<M; m++) {
			for(int t=0; t<instance.T; t++) {
				double maxOverK = -Double.MAX_VALUE;
				double[] updateValue = new double[K];
				double normalizer = 0;
				for(int k=0; k<K; k++) {					
					double sumThetaOverY = 0;
					for(int y=0; y<model.param.weights.vocabSize; y++) {
						sumThetaOverY += model.param.weights.get(m, k, y);
					}
					updateValue[k] = model.param.weights.get(m, k, instance.words[t][0]) - 0.5 * sumThetaOverY;
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
						
						sumY -= 2 * alpha.alpha * model.param.weights.get(m, k, y);
						MathUtils.check(sumY);
						updateValue[k] -= lambda * sumY;
						MathUtils.check(updateValue[k]);
					}
					normalizer += Math.exp(updateValue[k]);
					if(updateValue[k] > maxOverK) {
						maxOverK = updateValue[k];
					}					
				}
				//normalize
				for(int k=0; k<K; k++) {
					double oldValue = instance.varParamObs.shi[m][t][k];
					instance.varParamObs.shi[m][t][k] = updateValue[k] - maxOverK;
					//instance.varParamObs.shi[m][t][k] = updateValue[k] - Math.log(normalizer);
					MathUtils.check(instance.varParamObs.shi[m][t][k]);
					shiL1NormInstance += Math.abs(oldValue - instance.varParamObs.shi[m][t][k]);
				}
			}
			//TODO: decide if computing posteriors in bulk is better or after each layer
			//re-estimate state posteriors for this layer
			//instance.forwardBackwardList.get(m).doInference();				
		}	
		zeta.clearLambdaCache();
		shiL1NormAll += shiL1NormInstance;
	}
	
	public void optimizeCorpusParam() {
		int maxIter = 1;
		for(int iter=0; iter<maxIter; iter++) {
			//optimize zetas
			double zetaL1Norm = 0;
			for(int y=0; y<model.param.weights.vocabSize; y++) {
				double oldZeta = zeta.zeta[y];
				zeta.zeta[y] = 0;
				int totalT = 0;
				for(int i=0; i<Corpus.trainInstanceEStepSampleList.size(); i++) {
					Instance inst = Corpus.trainInstanceEStepSampleList.get(i);
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
				zeta.zeta[y] = zeta.zeta[y]/totalT;
				//value should not be negative because we still have to take squareroot
				if(zeta.zeta[y] <= 0) {
					System.err.println("zeta value <= 0 found, value=" + zeta.zeta[y]);
					//fix it
					zeta.zeta[y] = 1e-100;
				}
				zeta.zeta[y] = Math.sqrt(zeta.zeta[y]);
				MathUtils.check(zeta.zeta[y]);
				zetaL1Norm += Math.abs(zeta.zeta[y] - oldZeta);
			}
			//System.out.println("zetaL1Norm : " + zetaL1Norm);
		
			//optimize alpha
			zeta.createLambdaCache();
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
			zeta.clearLambdaCache();
			denominator = 2 * Corpus.trainInstanceEStepSampleList.numberOfTokens * denominator;
			double oldAlpha = alpha.alpha;
			alpha.alpha = numerator / denominator;
			MathUtils.check(alpha.alpha);
			//System.out.println("alphaL1Norm : " + Math.abs(oldAlpha - alpha.alpha));
		}
	}
}
