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
	int maxConsecutiveDecreaseLimit = 5;

	HMMParamBase expectedCounts;

	int lowerCount = 0; // number of times LL could not increase from previous
						// best
	int iterCount = 0;

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
		for (int n = 0; n < c.trainInstanceEStepSampleList.size(); n++) {
			Instance instance = c.trainInstanceEStepSampleList.get(n);
			instance.doInference(model);
			instance.forwardBackward.addToCounts(expectedCounts);
			LL += instance.forwardBackward.logLikelihood;
			//instance.createDecodedViterbiCache();
			instance.clearInference();
		}
		//MyArray.printExpTable(model.param.transition.get(0).count);
	}

	public void mStep() {
		// MyArray.printTable(expectedCounts.initial.count);
		// MyArray.printTable(expectedCounts.transition.count);
		// MyArray.printTable(expectedCounts.observation.count);
		model.updateFromCounts(expectedCounts);

		System.out.println("Mstep #tokens : " + c.trainInstanceMStepSampleList.numberOfTokens);
		// also update the log-linear model weights
		// maximize CLL of the data
		double[] initParams = MyArray.createVector(model.param.weights.weights);
		model.param.weights.weights = null;
		LogLinearWeightsOptimizable optimizable = new LogLinearWeightsOptimizable(initParams, c);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		boolean converged = false;
		try {
			converged = optimizer.optimize(5); //5 iters
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
			if(iterCount > 20) {
				Main.sampleSizeEStep = 2 * Main.sampleSizeEStep;
				Main.sampleSizeMStep = 2 * Main.sampleSizeMStep;
			}
			Timing oneIterEmTime = new Timing();
			//sample new train instances
			c.generateRandomTrainingEStepSample(Main.sampleSizeEStep);
			oneIterEmTime.start();
			LL = 0;
			// e-step
			eStepTime.start();
			Stats.totalFixes = 0;
			eStep();
			if (iterCount > 0) {
				System.out.format("LL %.2f Diff %.2f \t Iter %d \t Fixes: %d \t iter time %s\n",LL, (LL - bestOldLL), iterCount,Stats.totalFixes, eStepTime.stop());
			}
			if (isConverged()) {
				break;
			}
			// m-step
			c.generateRandomTrainingMStepSample(Main.sampleSizeMStep);
			mStep();
			System.out.format("iter EM time : %s\n" , oneIterEmTime.stop());			
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
