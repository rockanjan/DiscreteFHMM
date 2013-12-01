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

	double trainCll = 0;
	double devCll = 0;

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
		Corpus.trainInstanceEStepSampleList.updateExpectedCounts(model, expectedCounts);		
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
		trainCll = jointOptimizatble.getValue();
		trainCll = trainCll / Corpus.trainInstanceMStepSampleList.numberOfTokens; //per token CLL
		System.out.println("train joint CLL = " + trainCll);
		
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
	
	/*
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
		//arithmetic mean
		model.param.weightsClass.weights = MathUtils.weightedAverageofLog(optimizable.getParameterMatrix(),
				model.param.weightsClass.weights,
				adaptiveWeight);
	}
	*/

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
			double memoryUsed = 1.0 * (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024;
			System.out.format("Memory Used = %.2f MB\n",memoryUsed);
			//sample new train instances
			c.generateRandomTrainingEStepSample(Config.sampleSizeEStep, iterCount);
			// e-step
			eStepTime.start();
			oneIterEmTime.start();
			eStep();
			System.out.println("E-step time: " + eStepTime.stop());
			// m-step
			c.generateRandomTrainingMStepSample(Config.sampleSizeMStep);
			mStepTime.start();
			double oldTrainCll = trainCll;
			mStep();
			System.out.println("M-step time:" + mStepTime.stop());
			Stats.totalFixes = 0;
			StringBuffer display = new StringBuffer();
			display.append("Iter="+iterCount);
			
			boolean isConvergedCll = false;
			//Check on validation
			if(Corpus.devInstanceList != null && iterCount % Config.convergenceIterInterval == 0) {
				double[] paramsWord = MyArray.createVector(model.param.weights.weights);
				double[] paramsClass = MyArray.createVector(model.param.weightsClass.weights);
				double[] paramsJoint = MyArray.joinVectors(paramsWord, paramsClass);
				c.generateRandomDevSample(Config.sampleDevSize);
				double oldDevCll = devCll;
				//get devCll, but first, need to do inference
				expectedCounts.initializeZerosInitialAndTransition(); //mstep already complete
				Corpus.devInstanceSampleList.updateExpectedCounts(model, expectedCounts); //update expectations
				devCll = Corpus.devInstanceSampleList.getCLLJoint(paramsJoint);
				devCll = devCll / Corpus.devInstanceSampleList.numberOfTokens;
				double devCllDiff = devCll - oldDevCll;
				if(isConvergedCll(devCllDiff)) {
					isConvergedCll = true;
				}
				display.append(String.format(" devCll=%.5f dDiff=%.5f ", devCll, devCllDiff));
				if(Math.pow(model.nrStates, model.nrLayers) <= 100) {
                    System.out.println("Checking Test Perplexity");
                    Main.checkTestPerplexity();
                }
			}
			double trainCllDiff = trainCll - oldTrainCll;
			if(Corpus.devInstanceList == null) {
				if(isConvergedCll(trainCllDiff)) {
					isConvergedCll = true;
				}
			}
			model.updateL1Diff();
			display.append(String.format("trainCll %.5f tDiff %.5f initDiff %.6f transDiff %.6f iter time %s\n",
					trainCll, trainCllDiff, model.param.l1DiffInitialMax, model.param.l1DiffTransitionMax, oneIterEmTime.stop()));
			System.out.println(display.toString());
			
			if(iterCount % Config.modelSaveInterval == 0 && iterCount > 0) {
				model.saveModel(iterCount); //save every iteration
			}
			//also save the file with the itercount
			LastIter.write(iterCount);
			//check convergence
			if(model.isParamConverged() && isConvergedCll) {
				convergeCount++;
				if(convergeCount > Config.maxConsecutiveConvergeLimit) {
					System.out.println("params and CLL converged. Saving the final model");
					model.saveModel();
					break;
				}
			} else {
				convergeCount = 0; //reset
			}
			
		}
		System.out.println("Total EM Time : " + totalEMTime.stop());
	}
	
	public boolean isConvergedCll(double cllDiff) {
		if(cllDiff < 0) {
			return false;
		}
		if(Math.abs(cllDiff) > Config.precision) {
			return false;
		}
		return true;
	}
}
