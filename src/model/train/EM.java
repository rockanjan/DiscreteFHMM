package model.train;

import java.io.FileNotFoundException;

import model.HMMBase;
import model.HMMType;
import model.param.HMMParamBase;
import model.param.HMMParamFinalState;
import model.param.HMMParamNoFinalState;
import model.param.HMMParamNoFinalStateLog;
import program.Main;
import util.MathUtils;
import util.MyArray;
import util.Stats;
import util.Timing;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;
import config.Config;
import config.LastIter;
import corpus.Corpus;
import corpus.WordClass;

public class EM {
	int numIter;
	Corpus c;
	HMMBase model;

	double bestOldLL = -Double.MAX_VALUE;
	double LL = 0;

	double bestOldLLDev = -Double.MAX_VALUE;
	double devLL = 0;

	HMMParamBase expectedCounts;

	int convergeCount = 0;
	int lowerCount = 0; // number of times LL could not increase from previous
						// best
	public int iterCount = 0;
	double adaptiveWeight;

	public EM(int numIter, Corpus c, HMMBase model) {
		this.numIter = numIter;
		this.c = c;
		this.model = model;
	}

	public void setAdaptiveWeight() {
		//fraction of data
		double f = 1.0 * Corpus.trainInstanceEStepSampleList.numberOfTokens / Corpus.trainInstanceList.numberOfTokens;
		if(f == 1) {
			System.out.println("fraction = 1, all dataset used");
			adaptiveWeight = 1.0;
		} else {
			//standard adaptiveWeight technique
			//adaptiveWeight = (t0 + iterCount)^(-alpha)
			adaptiveWeight = Math.pow((Config.t0 + iterCount), - Config.alpha);
		}

		/*
		 * my approach based on iterations and fraction of samples selected
		 */
		/*
		//where a= f * maxIter / (1-f), where f is fraction of data used for trainining
		double f = 1.0 * Corpus.trainInstanceEStepSampleList.numberOfTokens / Corpus.trainInstanceList.numberOfTokens;
		double a;
		if(f == 1) {
			adaptiveWeight = 1;
		} else {
			a = f * numIter / (1-f);
			adaptiveWeight = a / (a + iterCount);
		}
		*/

	}

	public void eStep() {
		//expectedCounts.initializeZeros();
		expectedCounts.initializeZerosInitialAndTransition();
		System.out.format("Estep #sentences = %d, #tokens = %d\n",
				Corpus.trainInstanceEStepSampleList.size(),
				Corpus.trainInstanceEStepSampleList.numberOfTokens);
		LL = Corpus.trainInstanceEStepSampleList.updateExpectedCounts(model, expectedCounts);
		LL = LL/Corpus.trainInstanceEStepSampleList.numberOfTokens; //per token
	}

	public void mStep() {
		setAdaptiveWeight();
		System.out.println("Iter : " + iterCount + " adaptiveWeight : " + adaptiveWeight);

		System.out.format("Mstep #sentences = %d, #tokens = %d\n",
				Corpus.trainInstanceMStepSampleList.size(),
				Corpus.trainInstanceMStepSampleList.numberOfTokens);
		/*
		//Corpus.cacheFrequentConditionals();
		System.out.println("training words...");
		trainLBFGS();
		System.out.println("training classes...");
		trainLBFGSClass();
		*/
		trainLBFGSJoint();
		//Corpus.clearFrequentConditionals();
		model.updateFromCountsWeighted(expectedCounts, adaptiveWeight);
		//model.updateFromCounts(expectedCounts); //unweighted
		Corpus.trainInstanceEStepSampleList.clearPosteriorProbabilities();
		Corpus.trainInstanceEStepSampleList.clearDecodedStates();
		model.updateL1Diff();
		System.out.println("l1MaxDiffInit = " + model.param.l1DiffInitialMax + " l1MaxDiffTrans = " + model.param.l1DiffTransitionMax);
	}
	
	public void trainLBFGSJoint() {
		double[] initParamsWord = MyArray.createVector(model.param.weights.weights);
		double[] initParamsClass = MyArray.createVector(model.param.weightsClass.weights);
		double[] initParamsJoint = MyArray.joinVectors(initParamsWord, initParamsClass);
		CLLTrainerJoint jointOptimizatble = new CLLTrainerJoint(initParamsJoint, c);
		
		Optimizer jointOptimizer = new LimitedMemoryBFGS(jointOptimizatble);
		boolean converged = false;
		
		try {
			converged = jointOptimizer.optimize(Config.mStepIter);
		} catch (IllegalArgumentException e) {
			System.out.println("optimization threw exception: IllegalArgument");
		} catch (OptimizationException oe) {
			System.out.println("optimization threw OptimizationException");
		}
		System.out.println("Converged = " + converged);
		System.out.println("joint Gradient call count = " + jointOptimizatble.gradientCallCount);
		double cll = jointOptimizatble.getValue();
		cll = cll / Corpus.trainInstanceMStepSampleList.numberOfTokens; //per token CLL
		System.out.println("joint CLL = " + cll);
		
		//split params and assign it to the model
		double[][] splittedParams = MyArray.splitVector(jointOptimizatble.getParameterVector(), initParamsWord.length);
		double[][] wordParamMatrix = MyArray.createMatrix(splittedParams[0], c.corpusVocab.get(0).vocabSize);
		double[][] classParamMatrix = MyArray.createMatrix(splittedParams[1], WordClass.numClusters);
		model.param.weights.weights = MathUtils.weightedAverageofLog(wordParamMatrix,
				model.param.weights.weights,
				adaptiveWeight);
		
		model.param.weightsClass.weights = MathUtils.weightedAverageofLog(classParamMatrix,
				model.param.weightsClass.weights,
				adaptiveWeight);
	}

	public void trainLBFGS() {
		// maximize CLL of the data
		double[] initParams = MyArray.createVector(model.param.weights.weights);
		CLLTrainer optimizable = new CLLTrainer(initParams, c);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		boolean converged = false;
		//optimizable.checkGradientComputation();
		try {
			converged = optimizer.optimize(Config.mStepIter);
		} catch (IllegalArgumentException e) {
			System.out.println("optimization threw exception: IllegalArgument");
		} catch (OptimizationException oe) {
			System.out.println("optimization threw OptimizationException");
		}
		System.out.println("Converged = " + converged);
		System.out.println("Gradient call count = " + optimizable.gradientCallCount);
		double cll = optimizable.getValue();
		cll = cll / Corpus.trainInstanceMStepSampleList.numberOfTokens; //per token CLL
		System.out.println("CLL = " + cll);
		//model.param.weights.weights = optimizable.getParameterMatrix(); //unweighted
		
		//geometric mean
		/*
		model.param.weights.weights = MathUtils.weightedAverageMatrix(optimizable.getParameterMatrix(),
				model.param.weights.weights,
				adaptiveWeight);
		*/
		//arithmetic mean
		model.param.weights.weights = MathUtils.weightedAverageofLog(optimizable.getParameterMatrix(),
				model.param.weights.weights,
				adaptiveWeight);
	}
	
	public void trainLBFGSClass() {
		double[] initParams = MyArray.createVector(model.param.weightsClass.weights);
		CLLTrainerClass optimizable = new CLLTrainerClass(initParams, c);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		boolean converged = false;
		//optimizable.checkGradientComputation();
		try {
			converged = optimizer.optimize(Config.mStepIter);
		} catch (IllegalArgumentException e) {
			System.out.println("optimization threw exception: IllegalArgument");
		} catch (OptimizationException oe) {
			System.out.println("optimization threw OptimizationException");
		}
		System.out.println("Converged = " + converged);
		System.out.println("Gradient call count = " + optimizable.gradientCallCount);
		double cll = optimizable.getValue();
		cll = cll / Corpus.trainInstanceMStepSampleList.numberOfTokens; //per token CLL
		System.out.println("class CLL = " + cll);
		//model.param.weights.weights = optimizable.getParameterMatrix(); //unweighted
		
		//geometric mean
		/*
		model.param.weights.weights = MathUtils.weightedAverageMatrix(optimizable.getParameterMatrix(),
				model.param.weights.weights,
				adaptiveWeight);
		*/
		//arithmetic mean
		model.param.weightsClass.weights = MathUtils.weightedAverageofLog(optimizable.getParameterMatrix(),
				model.param.weightsClass.weights,
				adaptiveWeight);
	}

	public void start() throws FileNotFoundException {
		if (model.hmmType == HMMType.WITH_NO_FINAL_STATE) {
			expectedCounts = new HMMParamNoFinalState(model);
		} else if (model.hmmType == HMMType.WITH_FINAL_STATE) {
			expectedCounts = new HMMParamFinalState(model);
		} else if (model.hmmType == HMMType.LOG_SCALE) {
			expectedCounts = new HMMParamNoFinalStateLog(model);
		}
		System.out.println("Starting EM");
		Timing totalEMTime = new Timing();
		totalEMTime.start();
		Timing eStepTime = new Timing();
		Timing mStepTime = new Timing();
		Timing oneIterEmTime = new Timing();
		for (iterCount = Main.lastIter + 1; iterCount < numIter; iterCount++) {
			//sample new train instances
			c.generateRandomTrainingEStepSample(Config.sampleSizeEStep, iterCount);
			LL = 0;
			// e-step
			eStepTime.start();
			oneIterEmTime.start();
			eStep();
			System.out.println("E-step time: " + eStepTime.stop());
			double trainPerplexityJoint = Math.pow(2, -LL/Math.log(2));
			System.out.println("Train perplexity : " + trainPerplexityJoint);

			double diff = LL - bestOldLL;
			// m-step
			c.generateRandomTrainingMStepSample(Config.sampleSizeMStep);
			mStepTime.start();
			mStep();
			System.out.println("M-step time:" + mStepTime.stop());
			Stats.totalFixes = 0;
			StringBuffer display = new StringBuffer();
			if(Corpus.devInstanceList != null && iterCount % Config.convergenceIterInterval == 0) {
				c.generateRandomDevSample(Config.sampleDevSize);
				System.out.println(String.format("Dev #sentence=%d, #tokens=%d", Corpus.devInstanceSampleList.size(), Corpus.devInstanceSampleList.numberOfTokens));
				expectedCounts.initializeZerosInitialAndTransition(); //mstep already complete
				devLL = Corpus.devInstanceSampleList.updateExpectedCounts(model, expectedCounts);
				devLL = devLL / Corpus.devInstanceSampleList.numberOfTokens;
				double devPerplexityJoint = Math.pow(2, -devLL/Math.log(2));
				double devPerplexityCLL = Math.pow(2, -Corpus.devInstanceSampleList.getCLL(model.param.weights.weights)/Math.log(2)/Corpus.devInstanceSampleList.numberOfTokens);
				double devPerplexityLL = Math.pow(2, -Corpus.devInstanceSampleList.LL /Math.log(2)/Corpus.devInstanceSampleList.numberOfTokens);
				System.out.println("Dev Perplexity LL = " + devPerplexityLL + " Joint = " + devPerplexityJoint + " CLL = " + devPerplexityCLL);
				double devDiff = devLL - bestOldLLDev;
				if(iterCount > 0) {
					display.append(String.format("DevLL %.5f devDiff %.5f ", devLL, devDiff));
				}
				if(Math.pow(model.nrStates, model.nrLayers) <= 100) {
                                        System.out.println("Checking Test Perplexity");
                                        Main.checkTestPerplexity();
                                }
				if(isConverged()) {
					break;
				}
			}
			if (iterCount > 0) {
				display.append(String.format("obj %.5f Diff %.5f \t Iter %d \t Fixes: %d \t iter time %s\n",LL, diff, iterCount,Stats.totalFixes, oneIterEmTime.stop()));
			}
			//only check if not check in dev done (because dev was null)
			if(Corpus.devInstanceList == null) {
				if (isConverged()) {
					break;
				}
			}
			if(bestOldLL < LL) {
				bestOldLL = LL;
			}
			model.saveModel(iterCount); //save every iteration
			//also save the file with the itercount
			LastIter.write(iterCount);
			
			System.out.println(display.toString());
		}
		if(Corpus.testInstanceList != null) {
			double testLL = Corpus.testInstanceList.updateExpectedCounts(model, expectedCounts);
			testLL = testLL / Corpus.testInstanceList.numberOfTokens;
			double testPerplexity = Math.pow(2, -testLL/Math.log(2));
			System.out.println("Test Perplexity : " + testPerplexity);
		}
		System.out.println("Total EM Time : " + totalEMTime.stop());
	}

	public boolean isConverged() {
		//if no dev data, use training data itself for convergence test
		if(Corpus.devInstanceList == null) {
			devLL = LL;
			bestOldLLDev = bestOldLL;
		}

		double decreaseRatio = (devLL - bestOldLLDev) / Math.abs(bestOldLLDev);
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
		if (devLL < bestOldLLDev) {
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
			bestOldLLDev= devLL;
			return false;
		}
	}
}
