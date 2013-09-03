package model.train;

import config.Config;
import program.Main;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;
import model.HMMBase;
import model.HMMType;
import model.param.HMMParamBase;
import model.param.HMMParamFinalState;
import model.param.HMMParamNoFinalState;
import model.param.HMMParamNoFinalStateLog;

import util.MathUtils;
import util.MyArray;
import util.Stats;
import util.Timing;
import corpus.Corpus;

public class EM {
	int numIter;
	Corpus c;
	HMMBase model;

	double bestOldLL = -Double.MAX_VALUE;
	double LL = 0;
	HMMParamBase expectedCounts;

	int convergeCount = 0;
	int lowerCount = 0; // number of times LL could not increase from previous
						// best
	int iterCount = 0;
	double adaptiveWeight;
	
	public EM(int numIter, Corpus c, HMMBase model) {
		this.numIter = numIter;
		this.c = c;
		this.model = model;
	}

	public void eStep() {
		if (model.hmmType == HMMType.WITH_NO_FINAL_STATE) {
			expectedCounts = new HMMParamNoFinalState(model);
		} else if (model.hmmType == HMMType.WITH_FINAL_STATE) {
			expectedCounts = new HMMParamFinalState(model);
		} else if (model.hmmType == HMMType.LOG_SCALE) {
			expectedCounts = new HMMParamNoFinalStateLog(model);
		}
		expectedCounts.initializeZeros();
		System.out.format("Estep #sentences = %d, #tokens = %d\n", 
				Corpus.trainInstanceEStepSampleList.size(), 
				Corpus.trainInstanceEStepSampleList.numberOfTokens);
		LL = Corpus.trainInstanceEStepSampleList.updateExpectedCounts(model, expectedCounts);
	}

	public void mStep() {
		//adaptiveWeight based on : Online EM paper (Liang and Klein)
		//adaptiveWeight = Math.pow((1.0 * iterCount + 2.0), - Config.alpha);
		
		//adaptive weight w = a / (a+iter), 
		//where a= f * maxIter / (1-f), where f is fraction of data used for trainining
		double f = 1.0 * Corpus.trainInstanceEStepSampleList.numberOfTokens / Corpus.trainInstanceList.numberOfTokens;
		double a;
		if(f == 1) {
			adaptiveWeight = 1;
		} else {
			a = f * numIter / (1-f);
			adaptiveWeight = a / (a + iterCount);
		}
		System.out.println("Iter : " + iterCount + " frac : " + f + " adaptiveWeight : " + adaptiveWeight);
		
		System.out.format("Mstep #sentences = %d, #tokens = %d\n", 
				Corpus.trainInstanceMStepSampleList.size(), 
				Corpus.trainInstanceMStepSampleList.numberOfTokens);
		Corpus.cacheFrequentConditionals();
		trainLBFGS();
		Corpus.clearFrequentConditionals();
		model.updateFromCountsWeighted(expectedCounts, adaptiveWeight);
		//model.updateFromCounts(expectedCounts); //unweighted
		Corpus.trainInstanceEStepSampleList.clearPosteriorProbabilities();
		Corpus.trainInstanceEStepSampleList.clearDecodedStates();
	}
	
	public void trainLBFGS() {
		// maximize CLL of the data
		double[] initParams = MyArray.createVector(model.param.weights.weights);
		CLLTrainer optimizable = new CLLTrainer(initParams, c);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		boolean converged = false;
		try {
			converged = optimizer.optimize(Config.mStepIter);
		} catch (IllegalArgumentException e) {
			System.out.println("optimization threw exception: IllegalArgument");
		} catch (OptimizationException oe) {
			System.out.println("optimization threw OptimizationException");
		}
		System.out.println("Converged = " + converged);
		System.out.println("Gradient call count: " + optimizable.gradientCallCount);
		//model.param.weights.weights = optimizable.getParameterMatrix();
		model.param.weights.weights = MathUtils.weightedAverageMatrix(optimizable.getParameterMatrix(), 
				model.param.weights.weights, 
				adaptiveWeight);
	}

	public void start() {
		System.out.println("Starting EM");
		Timing totalEMTime = new Timing();
		totalEMTime.start();
		Timing eStepTime = new Timing();
		Timing oneIterEmTime = new Timing();
		for (iterCount = 0; iterCount < numIter; iterCount++) {
			//sample new train instances
			c.generateRandomTrainingEStepSample(Config.sampleSizeEStep);
			LL = 0;
			// e-step
			eStepTime.start();
			eStep();
			System.out.println("E-step time: " + eStepTime.stop());
			double diff = LL - bestOldLL;
			if (iterCount > 0) {
				System.out.format("LL %.2f Diff %.2f \t Iter %d \t Fixes: %d \t iter time %s\n",LL, diff, iterCount,Stats.totalFixes, oneIterEmTime.stop());
			}
			if (isConverged()) {
				break;
			}
			oneIterEmTime.start();
			// m-step
			c.generateRandomTrainingMStepSample(Config.sampleSizeMStep);
			mStep();
			Stats.totalFixes = 0;
			if(iterCount % 5 == 0 && c.devInstanceList != null) {
				System.out.println("Dev LL : " + c.devInstanceList.getLL(model));
			}
			model.saveModel(iterCount); //save every iteration
		}
		System.out.println("Total EM Time : " + totalEMTime.stop());
	}

	public boolean isConverged() {
		double decreaseRatio = (LL - bestOldLL) / Math.abs(bestOldLL);
		// System.out.println("Decrease Ratio: %.5f " + decreaseRatio);
		if (Config.precision > decreaseRatio && decreaseRatio > 0) {
			convergeCount++;
			if(convergeCount > Config.maxConsecutiveConvergeLimit) {
				System.out.println("Converged. Saving the final model");
				model.saveModel();
				return true;
			}
		}
		convergeCount = 0;
		if (LL < bestOldLL) {
			if (lowerCount == 0) {
				// cache the best model so far
				System.out.println("Caching the best model so far");
				if (model.bestParam != null) {
					model.bestParam.cloneFrom(model.param);
				}				
			}
			lowerCount++;
			if (lowerCount == Config.maxConsecutiveDecreaseLimit) {
				System.out.format("Saying Converged: LL could not increase for %d consecutive iterations\n",
						Config.maxConsecutiveDecreaseLimit);
				if (model.bestParam != null) {
					model.param.cloneFrom(model.bestParam);
				}
				return true;
			}
			return false;
		} else {
			lowerCount = 0;
			bestOldLL = LL;
			return false;
		}
	}
}
