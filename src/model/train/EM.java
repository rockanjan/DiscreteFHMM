package model.train;

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
import corpus.Instance;

public class EM {

	int numIter;
	Corpus c;
	HMMBase model;

	double bestOldLL = -Double.MAX_VALUE;
	double LL = 0;

	// convergence criteria
	double precision = 1e-6;
	int maxConsecutiveDecreaseLimit = 20;

	HMMParamBase expectedCounts;

	int lowerCount = 0; // number of times LL could not increase from previous
						// best
	int iterCount = 0;
	
	int mStepIter = 20; //initial
	
	

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
		System.out.println("Estep #tokens : " + c.trainInstanceEStepSampleList.numberOfTokens);
		LL = c.trainInstanceEStepSampleList.updateExpectedCounts(model, expectedCounts);		
	}

	public void mStep() {
		System.out.println("Mstep #tokens : " + c.trainInstanceMStepSampleList.numberOfTokens);
		//trainPerceptron();
		//trainLogLinearOptimization();
		trainAveragedPerceptron();
		//trainSgd();
		model.updateFromCounts(expectedCounts);
	}
	
	public void trainSgd() {
		SgdTrainer sgd = new SgdTrainer(c);
		sgd.train(model.param.weights.weights, mStepIter);
	}
	
	public void trainAveragedPerceptron() {
		//first do the viterbi decoding
		Timing decodeTiming = new Timing();
		decodeTiming.start();
		model.param.expWeightsCache = MathUtils.expArray(model.param.weights.weights);
		for(int n=0; n<c.trainInstanceMStepSampleList.size(); n++) {
			Instance instance = c.trainInstanceMStepSampleList.get(n);
			//instance.observationCache = null;
			//instance.doInference(model);
			instance.createDecodedViterbiCache();
		}
		System.out.println("M-step Decode time : " + decodeTiming.stop());
		AveragedPerceptronTrainerViterbi pt = new AveragedPerceptronTrainerViterbi(c);
		pt.train(model.param.weights.weights, mStepIter);
		model.param.expWeightsCache = null;
	}
	
	public void trainPerceptron() {
		PerceptronTrainer pt = new PerceptronTrainer(c);
		pt.train(model.param.weights.weights, mStepIter);
				
	}
	
	public void trainLogLinearOptimization() {
		// maximize CLL of the data
		double[] initParams = MyArray.createVector(model.param.weights.weights);
		model.param.weights.weights = null;
		CLLTrainer optimizable = new CLLTrainer(initParams, c);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		boolean converged = false;
		try {
			converged = optimizer.optimize(mStepIter);
		} catch (IllegalArgumentException e) {
			System.out.println("optimization threw exception: IllegalArgument");
		} catch (OptimizationException oe) {
			System.out.println("optimization threw OptimizationException");
		}
		System.out.println("Gradient call count: " + optimizable.gradientCallCount);
		model.param.weights.weights = optimizable.getParameterMatrix();
		
	}

	public void start() {
		System.out.println("Starting EM");
		Timing totalEMTime = new Timing();
		totalEMTime.start();
		Timing eStepTime = new Timing();
		//c.trainInstanceSampleList = c.trainInstanceList;
		for (iterCount = 0; iterCount < numIter; iterCount++) {
			Timing oneIterEmTime = new Timing();
			//sample new train instances
			c.generateRandomTrainingEStepSample(Main.sampleSizeEStep);
			oneIterEmTime.start();
			LL = 0;
			// e-step
			eStepTime.start();
			Stats.totalFixes = 0;
			eStep();
			System.out.println("E-step time: " + eStepTime.stop());
			double diff = LL - bestOldLL;
			if (isConverged()) {
				break;
			}
			/*
			if(iterCount > 2) {
				mStepIter = 5;			
			}
			
			if(iterCount > 30 && iterCount % 2 == 0) {
				mStepIter += 1;
			}
			*/
			// m-step
			c.generateRandomTrainingMStepSample(Main.sampleSizeMStep);
			mStep();
			if (iterCount > 0) {
				System.out.format("LL %.2f Diff %.2f \t Iter %d \t Fixes: %d \t iter time %s\n",LL, diff, iterCount,Stats.totalFixes, oneIterEmTime.stop());
			}			
		}
		System.out.println("Total EM Time : " + totalEMTime.stop());
	}

	public boolean isConverged() {
		
		double decreaseRatio = (LL - bestOldLL) / Math.abs(bestOldLL);
		// System.out.println("Decrease Ratio: %.5f " + decreaseRatio);
		if (precision > decreaseRatio && decreaseRatio > 0) {
			System.out.println("Converged. Saving the final model");
			model.saveModel(Main.currentRecursion);
			return true;
		}

		if (LL < bestOldLL) {
			//increase the number of examples in M-step
			/*
			if(Main.sampleSizeMStep < 5000) {
				Main.sampleSizeMStep += 250;
			}
			*/
			if (lowerCount == 0) {
				// cache the best model so far
				System.out.println("Caching the best model so far");
				if (model.bestParam != null) {
					model.bestParam.cloneFrom(model.param);
				}				
			}
			lowerCount++;
			if (lowerCount == maxConsecutiveDecreaseLimit) {
				System.out.format("Saying Converged: LL could not increase for %d consecutive iterations\n",maxConsecutiveDecreaseLimit);
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
