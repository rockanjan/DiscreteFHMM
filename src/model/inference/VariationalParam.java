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
	public VariationalParamZeta zeta;
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
		zeta = new VariationalParamZeta(V, T);		
		zeta.initializeRandom();
	}
	
	public void optimize() {
		optimizeParamObs();
		optimizeZeta();
		optimizeAlpha();		
	}
	
	public void optimizeParamObs() {
		//optimize shi's
		double shiL1NormInstance = 0;
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
						double lambda = zeta.lambdaZeta(y, t);
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
						
						sumY -= 2 * alpha.alpha[t] * model.param.weights.get(m, k, y);
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
					double oldValue = varParamObs.shi[m][t][k];
					varParamObs.shi[m][t][k] = updateValue[k] - maxOverK;
					//instance.varParamObs.shi[m][t][k] = updateValue[k] - Math.log(normalizer);
					MathUtils.check(varParamObs.shi[m][t][k]);
					shiL1NormInstance += Math.abs(oldValue - varParamObs.shi[m][t][k]);
				}
			}
			//TODO: decide if computing posteriors in bulk is better or after each layer
			//re-estimate state posteriors for this layer
			//instance.forwardBackwardList.get(m).doInference();				
		}	
		InstanceList.shiL1NormAll += shiL1NormInstance;		
	}
	
	public void optimizeZeta() {
		double zetaL1Norm = 0;
		for(int y=0; y<model.param.weights.vocabSize; y++) {
			for(int t=0; t<T; t++) {
				double oldZeta = zeta.zeta[y][t];
				zeta.zeta[y][t] = Math.pow(alpha.alpha[t],2);
				for(int m=0; m<M; m++) {
					for(int n=0; n<M; n++) {
						if(m==n) continue;
						double dotThetaMStateM = MathUtils.dot(model.param.weights.getStateVector(m, y),
								instance.forwardBackwardList.get(m).posterior[t]);
						double dotThetaNStateN =  MathUtils.dot(model.param.weights.getStateVector(n, y),
								instance.forwardBackwardList.get(n).posterior[t]);
						zeta.zeta[y][t] += dotThetaMStateM * dotThetaNStateN;
					}
					double thetaMDiagStateMthetaM = MathUtils.vectorTransposeMatrixVector(
							model.param.weights.getStateVector(m, y), 
							MathUtils.diag(instance.forwardBackwardList.get(m).posterior[t]), 
							model.param.weights.getStateVector(m, y));
					zeta.zeta[y][t] += thetaMDiagStateMthetaM;
					double dotStateMThetaM = MathUtils.dot(
							instance.forwardBackwardList.get(m).posterior[t],
							model.param.weights.getStateVector(m, y)
							);
					zeta.zeta[y][t] -= 2 * alpha.alpha[t] * dotStateMThetaM;				
				}
				//value should not be negative because we still have to take squareroot
				if(zeta.zeta[y][t] <= 0) {
					System.err.println("zeta value <= 0 found, value=" + zeta.zeta[y][t]);
					//fix it
					zeta.zeta[y][t] = 1e-100;
				}
				zeta.zeta[y][t] = Math.sqrt(zeta.zeta[y][t]);
				MathUtils.check(zeta.zeta[y][t]);
				zetaL1Norm += Math.abs(zeta.zeta[y][t] - oldZeta);
			}
		}
		InstanceList.zetaL1NormAll += zetaL1Norm;
	}
	
	public void optimizeAlpha() {
		double alphaL1Norm = 0;
		for(int t=0; t<T; t++) {
			double sumY = 0;
			double sumLambdaY = 0;
			for(int y=0; y<model.param.weights.vocabSize; y++) {
				double sumM = 0;
				for(int m=0; m<M; m++) {
					double dotStateMThetaM = MathUtils.dot(
							instance.forwardBackwardList.get(m).posterior[t],
							model.param.weights.getStateVector(m, y)
							);
					sumM += dotStateMThetaM; 
				}
				double lambda = zeta.lambdaZeta(y, t); 
				sumY += lambda * sumM;
				sumLambdaY += lambda;
			}	
			double numerator = 0.5 * V - 1 + 2 * sumY;
			double denominator = 2 * sumLambdaY;
			double oldAlpha = alpha.alpha[t];
			alpha.alpha[t] = numerator / denominator;
			alphaL1Norm += Math.abs(alpha.alpha[t] - oldAlpha);
			MathUtils.check(alpha.alpha[t]);
		}
		InstanceList.alphaL1NormAll += alphaL1Norm;
	}
}
