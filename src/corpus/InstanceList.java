package corpus;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import config.Config;
import model.HMMBase;
import model.inference.VariationalParam;
import model.param.HMMParamBase;
import model.param.LogLinearWeights;
import util.MathUtils;
import util.Timing;

public class InstanceList extends ArrayList<Instance> {
	/*
	public static double shiL1NormAll=0;
	public static double alphaL1NormAll=0;
	*/
	public static double expectationL1NormAll=0;
	public static double expectationL1NormMax=0;
	
	//locks used for threads
	private static final Object gradientLockSoft = new Object();
	private static final Object cllLockSoft = new Object();
	private static final Object variationalLock = new Object();
	
	public double LL = 0;
	public double jointObjective = 0;
	double cll = 0;
	double gradient[][];
	
	private static final long serialVersionUID = -2409272084529539276L;
	public int numberOfTokens;	
	
	public InstanceList() {
		super();		
	}
	
	static public Map<String, Double> featurePartitionCache;

	/*
	 * just to get the LL of the data
	 */
	public double getLL(HMMBase model) {
		throw new UnsupportedOperationException("Not yet implemented");				
	}
	
	public double getJointLL(Instance instance, HMMBase model) {
		double jointLL = 0;
		//for t=0;
		for(int l=0; l<model.nrLayers; l++) {
			jointLL += model.param.initial.get(l).get(instance.decodedStates[l][0], 0);			
		}
		jointLL += instance.getObservationProbabilityUsingLLModel(0);
		
		for(int t=1; t<instance.T; t++) {
			for(int l=0; l<model.nrLayers; l++) {
				jointLL += model.param.transition.get(l).get(instance.decodedStates[l][t], instance.decodedStates[l][t-1]);
			}
			jointLL += instance.getObservationProbabilityUsingLLModel(t);
		}
		return jointLL;
	}
	
	/*
	 * called by the E-step of EM. 
	 * Does inference, computes posteriors and updates expected counts
	 * returns LL of the corpus
	 */
	public double updateExpectedCounts(HMMBase model, HMMParamBase expectedCounts) {
		//cache expWeights for the model
		model.param.expWeights = model.param.weights.getCloneExp();
		
		featurePartitionCache = new ConcurrentHashMap<String, Double>();
		doVariationalInference(model);
		
		//decode the most likely states and compute joint likelihood to return
		double jointLL = 0;
		LL = 0;
		jointObjective = 0;
		for (int n = 0; n < this.size(); n++) {
			Instance instance = this.get(n);
			instance.doInference(model);
			LL += instance.logLikelihood;
			jointObjective += instance.jointObjective;
			for(int l=0; l<model.nrLayers; l++) {
				instance.forwardBackwardList.get(l).addToCounts(expectedCounts);
			}
			//instance.decode();
			//jointLL += getJointLL(instance, model);
			//instance.clearInference();
			instance.varParam = null;
		}
		//clear expWeights;				
		model.param.expWeights = null;
		featurePartitionCache = null;
		System.out.println("LL = " + (LL/numberOfTokens));
		return jointObjective;
	}
	
	public void clearPosteriorProbabilities() {
		for (int n = 0; n < this.size(); n++) {
			Instance instance = this.get(n);
			instance.posteriors = null;
		}
	}
	
	public void clearDecodedStates() {
		for (int n = 0; n < this.size(); n++) {
			Instance instance = this.get(n);
			instance.decodedStates = null;
		}
	}
	
	public void doVariationalInference(HMMBase model) {
		//optimize variational parameters
		for(int iter=0; iter < Config.variationalIter; iter++) {
			LL = 0;
			jointObjective = 0;
			expectationL1NormAll = 0;
			expectationL1NormMax = 0;
			Timing varIterTime = new Timing();
			varIterTime.start();
			
			//start parallel processing
			int divideSize = this.size() / Config.USE_THREAD_COUNT;
			List<VariationalWorker> threadList = new ArrayList<VariationalWorker>();
			int startIndex = 0;
			int endIndex = divideSize;		
			for(int i=0; i<Config.USE_THREAD_COUNT; i++) {
				VariationalWorker worker = new VariationalWorker(this, startIndex, endIndex, model);
				threadList.add(worker);
				worker.start();
				startIndex = endIndex;
				endIndex = endIndex + divideSize;			
			}
			//there might be some remaining
			VariationalWorker finalWorker = new VariationalWorker(this, startIndex, this.size(), model);
			finalWorker.start();
			threadList.add(finalWorker);
			//start all threads and wait for them to complete
			for(VariationalWorker worker : threadList) {
				try {
					worker.join();
				} catch (InterruptedException e) {				
					e.printStackTrace();
				}
				updateVariationalComputation(worker);
			}
			StringBuffer updateString = new StringBuffer();
			updateString.append("\tvar iter=" + iter);
			expectationL1NormAll = expectationL1NormAll/this.numberOfTokens/model.nrLayers/model.nrStates;
			LL = LL/this.numberOfTokens;
			jointObjective = jointObjective/this.numberOfTokens;
			updateString.append(String.format(" LL=%.5f time=%s", LL, varIterTime.stop()));
			updateString.append(String.format(" obj=%.5f", jointObjective));
			updateString.append(String.format(" l1avg=%.10f", expectationL1NormAll));
			updateString.append(String.format(" max=%.10f", expectationL1NormMax));
			System.out.println(updateString.toString());
			
			if(expectationL1NormMax < Config.variationalConvergence) {
				System.out.println("variational params converged");
				break;
			}
			
		}		
	}
	
	public void updateVariationalComputation(VariationalWorker worker) {
        synchronized (variationalLock) {
        	expectationL1NormAll += worker.localExpectationL1Norm;
			LL += worker.localLL;
			jointObjective += worker.localObjective;
			if(worker.localExpectationL1Max > expectationL1NormMax) {
				expectationL1NormMax = worker.localExpectationL1Max; 
			}
        }
    }
	
	private class VariationalWorker extends Thread{
		double localExpectationL1Norm = 0;
		double localExpectationL1Max = 0;
		double localLL = 0;
		double localObjective = 0;
		InstanceList instanceList;
		final int startIndex;
		final int endIndex;
		final HMMBase model;
		
		// [startIndex, endIndex) i.e. last index is not included
		public VariationalWorker(InstanceList instanceList, int startIndex, int endIndex, HMMBase model) {
			this.instanceList = instanceList;
			this.startIndex = startIndex;
			this.endIndex = endIndex;
			this.model = model;
		}
		
		@Override
		public void run() {
			localExpectationL1Norm = 0.0;
			localExpectationL1Max = 0.0;
			//instance level params
			for (int n = startIndex; n < endIndex; n++) {
				Instance instance = instanceList.get(n);
				instance.model = model;
				if(instance.varParam == null) {
					instance.varParam = new VariationalParam(model, instance);
					instance.doInference(model); //get expected states with unoptimized params
				}
				instance.varParam.optimize();
				instance.posteriorDifference = 0;
				instance.posteriorDifferenceMax = 0;
				instance.doInference(model);
				//instance.decode();
				//localLL += getJointLL(instance, model);
				localLL += instance.logLikelihood;
				localObjective += instance.jointObjective;
				localExpectationL1Norm += instance.posteriorDifference;
				if(instance.posteriorDifferenceMax > localExpectationL1Max) {
					localExpectationL1Max = instance.posteriorDifferenceMax; 
				}
				
			}
		}		
	}
	
	public void decode(HMMBase model) {
		//decode
		for (int n = 0; n < this.size(); n++) {
			Instance instance = this.get(n);
			instance.model = model;
			instance.decode();							
		}		
	}
	
	public double getCLL(double[][] parameterMatrix) {
		if(Config.USE_THREAD_COUNT < 2) {
			return getCLLSoft(parameterMatrix);
		} else {
			return getCLLSoftThreaded(parameterMatrix);
		}
	}
	
	public double[][] getGradient(double[][] parameterMatrix) {
		if(Config.USE_THREAD_COUNT < 2) {
			return getGradientSoft(parameterMatrix);
		} else {
			return getGradientSoftThreaded(parameterMatrix);
		}
		//return getGradientSoftNaive(parameterMatrix);
	}
	
	//returns the lowerbound value
	private double getCLLSoft(double[][] parameterMatrix) {
		Timing t = new Timing();
		cll = 0;
		double[][] expWeights = MathUtils.expArray(parameterMatrix);
		for (int n = 0; n < this.size(); n++) {
			Instance i = get(n);
			cll += i.getConditionalLogLikelihoodSoft(parameterMatrix, expWeights);
		}
		if(Config.displayDetail) {
			System.out.println("CLL computation time : " + t.stop());
		}
		return cll;
	}
	
	private double getCLLSoftThreaded(double[][] parameterMatrix) {
		Timing timing = new Timing();
		cll = 0.0;
		//start parallel processing
		int divideSize = this.size() / Config.USE_THREAD_COUNT;
		double[][] expWeights = MathUtils.expArray(parameterMatrix);
		List<CllSoftWorker> threadList = new ArrayList<CllSoftWorker>();
		int startIndex = 0;
		int endIndex = divideSize;		
		for(int i=0; i<Config.USE_THREAD_COUNT; i++) {
			CllSoftWorker worker = new CllSoftWorker(startIndex, endIndex, parameterMatrix, expWeights);
			threadList.add(worker);
			worker.start();
			startIndex = endIndex;
			endIndex = endIndex + divideSize;			
		}
		//there might be some remaining
		CllSoftWorker finalWorker = new CllSoftWorker(startIndex, this.size(), parameterMatrix, expWeights);
		finalWorker.start();
		threadList.add(finalWorker);
		//start all threads and wait for them to complete
		for(CllSoftWorker worker : threadList) {
			try {
				worker.join();
			} catch (InterruptedException e) {				
				e.printStackTrace();
			}
			updateCLLSoftComputation(worker);
		}
		if(Config.displayDetail) {
			System.out.println("CLL computation time : " + timing.stop());
		}
		return cll;
	}
	
	private class CllSoftWorker extends Thread{
		public double result;
		final int startIndex;
		final int endIndex;
		final double[][] expWeights;		
		final double[][] weights;
		
		// [startIndex, endIndex) i.e. last index is not included
		public CllSoftWorker(int startIndex, int endIndex, double[][] weights, double[][] expWeights) {
			this.startIndex = startIndex;
			this.endIndex = endIndex;
			this.expWeights = expWeights;
			this.weights = weights;
		}
		
		@Override
		public void run() {
			result = 0.0;
			for(int n=startIndex; n<endIndex; n++) {
				Instance i = get(n);
				result += i.getConditionalLogLikelihoodSoft(weights, expWeights);
			}
		}		
	}
	
	private void updateCLLSoftComputation(CllSoftWorker worker) {
	    synchronized (cllLockSoft) {
	    	cll += worker.result;
        }
    }
		
	private double[][] getGradientSoftThreaded(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		double[][] expParam = MathUtils.expArray(parameterMatrix);		
		gradient = new double[parameterMatrix.length][parameterMatrix[0].length];
		int divideSize = this.size() / Config.USE_THREAD_COUNT;
		List<GradientSoftWorker> threadList = new ArrayList<GradientSoftWorker>();
		int startIndex = 0;
		int endIndex = divideSize;		
		for(int i=0; i<Config.USE_THREAD_COUNT; i++) {
			GradientSoftWorker worker = new GradientSoftWorker(startIndex, endIndex, expParam);
			threadList.add(worker);
			worker.start();
			startIndex = endIndex;
			endIndex = endIndex + divideSize;			
		}
		//there might be some remaining
		GradientSoftWorker finalWorker = new GradientSoftWorker(startIndex, this.size(), expParam);
		finalWorker.start();
		threadList.add(finalWorker);
		//start all threads and wait for them to complete
		for(GradientSoftWorker worker : threadList) {
			try {
				worker.join();
			} catch (InterruptedException e) {				
				e.printStackTrace();
			}
			updateGradientSoftComputation(worker);
		}
		if(Config.displayDetail) {
			System.out.println("Gradient computation time : " + timing.stop());
		}
		return gradient;
	}
	
	private class GradientSoftWorker extends Thread{
		public double[][] gradientLocal;
		final int startIndex;
		final int endIndex;
		final double[][] expParam;
		public GradientSoftWorker(int startIndex, int endIndex, double[][] expParam) {
			this.startIndex = startIndex;
			this.endIndex = endIndex;
			this.expParam = expParam;			
		}
		
		@Override
		public void run() {
			gradientLocal = new double[expParam.length][expParam[0].length];
			for(int n=startIndex; n<endIndex; n++) {
				Instance instance = get(n);
				for(int t=0; t<instance.T; t++) {
					//positive evidence
					for(int m=0; m<Config.nrLayers; m++) {
						for(int k=0; k<Config.numStates; k++) {
							gradientLocal[instance.words[t][0]][LogLinearWeights.getIndex(m, k)] += instance.posteriors[m][t][k];						 
						}
					}
					int vocabSize = Corpus.corpusVocab.get(0).vocabSize;
					if(Config.VOCAB_SAMPLE_SIZE <= 0) { //exact
						//compute phi, variational param phi for this token
						double sumOverY = 0;
						for(int y=0; y<vocabSize; y++) {
							double prod = 1.0;
							for(int m=0; m<Config.nrLayers; m++) {
								double dot = 0;
								for(int k=0; k<Config.numStates; k++) {
									dot += instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)];
								}
								prod *= dot;
								MathUtils.check(prod);
								if(prod == 0) {
									throw new RuntimeException("underflow");
								}
							}
							sumOverY += prod;
						}
						double phi = 1.0 / sumOverY;
						
						for(int y=0; y<vocabSize; y++) {
							double dotProdOverAllLayers = 1.0; //to reduce complexity from O(m^2) to O(m)
							for(int m=0; m<Config.nrLayers; m++) {
								double dot = 0;
								for(int k=0; k<Config.numStates; k++) {
									dot += instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)];
								}
								dotProdOverAllLayers *= dot;
								MathUtils.check(dotProdOverAllLayers);
								if(dotProdOverAllLayers == 0) {
									throw new RuntimeException("underflow");
								}
							}
							//set them now
							for(int m=0; m<Config.nrLayers; m++) {
								double mLayerDot = 0.0;
								for(int l=0; l<Config.numStates; l++) {
									mLayerDot += instance.posteriors[m][t][l] * expParam[y][LogLinearWeights.getIndex(m, l)];
								}
								for(int k=0; k<Config.numStates; k++) {
									//compute the amount that must be multiplied to adjust from dotProdOverAllLayers
									double factorDifference = instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)] / mLayerDot;
									gradientLocal[y][LogLinearWeights.getIndex(m, k)] -= phi * dotProdOverAllLayers * factorDifference;
								}
							}
						}
					} else {
						//importance sampling
						double[][] a = new double[expParam.length][expParam[0].length];
						double b = 0.0;
						for(int s=0; s<Config.VOCAB_SAMPLE_SIZE; s++) {
							int y = Corpus.getRandomVocabItem();
							double dotProdOverAllLayers = 1.0;
							//compute numerator
							for(int m=0; m<Config.nrLayers; m++) {
								double dot = 0;
								for(int k=0; k<Config.numStates; k++) {
									dot += instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)];
								}
								dotProdOverAllLayers *= dot;
								MathUtils.check(dotProdOverAllLayers);
								if(dotProdOverAllLayers == 0) {
									throw new RuntimeException("underflow");
								}
							}
							double numerator = dotProdOverAllLayers;
							double denominator = Corpus.getProbability(y);
							double r = numerator / denominator;							
							for(int m=0; m<Config.nrLayers; m++) {
								double mLayerDot = 0.0;
								for(int l=0; l<Config.numStates; l++) {
									mLayerDot += instance.posteriors[m][t][l] * expParam[y][LogLinearWeights.getIndex(m, l)];
								}
								for(int k=0; k<Config.numStates; k++) {
									//calculate the function value
									double functionValue = instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)] / mLayerDot;
									a[y][LogLinearWeights.getIndex(m, k)] += r * functionValue;
									b += r;
								}
							}
						}
						//divide by b
						MathUtils.matrixElementWiseMultiplication(a, -1.0/b);
						//add to the gradient
						MathUtils.addMatrix(gradientLocal, a);
					}
				}
			}
		}
	}
	
	private void updateGradientSoftComputation(GradientSoftWorker worker) {
        synchronized (gradientLockSoft) {
        	MathUtils.addMatrix(gradient, worker.gradientLocal);
        }
    }
	
	private double[][] getGradientSoft(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		int vocabSize = Corpus.corpusVocab.get(0).vocabSize;
		double[][] expParam = MathUtils.expArray(parameterMatrix);
		gradient = new double[parameterMatrix.length][parameterMatrix[0].length];		
		for(int n=0; n<this.size(); n++) {
			Instance instance = get(n);
			for(int t=0; t<instance.T; t++) {
				for(int m=0; m<Config.nrLayers; m++) {
					for(int k=0; k<Config.numStates; k++) {
						gradient[instance.words[t][0]][LogLinearWeights.getIndex(m, k)] += instance.posteriors[m][t][k];						 
					}
				}
				//compute phi, variational param phi for this token
				double sumOverY = 0;
				for(int y=0; y<vocabSize; y++) {
					double dotProdOverAllLayers = 1.0;
					for(int m=0; m<Config.nrLayers; m++) {
						double dot = 0;
						for(int k=0; k<Config.numStates; k++) {
							dot += instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)];
						}
						dotProdOverAllLayers *= dot;
						MathUtils.check(dotProdOverAllLayers);
						if(dotProdOverAllLayers == 0) {
							throw new RuntimeException("underflow");
						}
					}
					sumOverY += dotProdOverAllLayers;
				}
				double phi = 1.0 / sumOverY;
				for(int y=0; y<vocabSize; y++) {
					double dotProdOverAllLayers = 1.0; //to reduce complexity from O(m^2) to O(m)
					for(int m=0; m<Config.nrLayers; m++) {
						double dot = 0;
						for(int k=0; k<Config.numStates; k++) {
							dot += instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)];
						}
						dotProdOverAllLayers *= dot;
						MathUtils.check(dotProdOverAllLayers);
						if(dotProdOverAllLayers == 0) {
							throw new RuntimeException("underflow");
						}
					}
					//set them now
					for(int m=0; m<Config.nrLayers; m++) {
						double mLayerDot = 0.0;
						for(int l=0; l<Config.numStates; l++) {
							mLayerDot += instance.posteriors[m][t][l] * expParam[y][LogLinearWeights.getIndex(m, l)];
						}
						for(int k=0; k<Config.numStates; k++) {
							//compute the amount that must be multiplied to adjust from dotProdOverAllLayers
							double factorDifference = instance.posteriors[m][t][k] * expParam[y][LogLinearWeights.getIndex(m, k)] / mLayerDot;
							gradient[y][LogLinearWeights.getIndex(m, k)] -= phi * dotProdOverAllLayers * factorDifference;
						}
					}
				}
				
			}
		}
		if(Config.displayDetail) {
			System.out.println("Gradient computation time : " + timing.stop());
		}
		return gradient;
	}
	
	
	//implemented according to the derivation in the paper
	private double[][] getGradientSoftNaive(double[][] parameterMatrix) {
		Timing timing = new Timing();
		timing.start();
		int vocabSize = Corpus.corpusVocab.get(0).vocabSize;
		double[][] expParam = MathUtils.expArray(parameterMatrix);
		double gradient[][] = new double[parameterMatrix.length][parameterMatrix[0].length];
		for(int m=0; m<Config.nrLayers; m++) {
			for(int y=0; y<vocabSize; y++) {
				for(int k=0; k<Config.numStates; k++) {
					for(int n=0; n<this.size(); n++) {
						Instance instance = get(n);
						for(int t=0; t<instance.T; t++) {
							if(instance.words[t][0] == y) {
								gradient[y][LogLinearWeights.getIndex(m, k)] += instance.posteriors[m][t][k];
							}
							double prod = 1.0;
							for(int p=0; p<Config.nrLayers; p++){
								if(p == m) {
									prod *= instance.posteriors[p][t][k] * expParam[y][LogLinearWeights.getIndex(p, k)];
								} else {
									double dot = 0.0;
									for(int l=0; l<Config.numStates; l++) {
										dot += instance.posteriors[p][t][l] * expParam[y][LogLinearWeights.getIndex(p, l)];
									}
									prod *= dot;
								}
								MathUtils.check(prod);
								if(prod == 0) {
									throw new RuntimeException("underflow");
								}
							}
							gradient[y][LogLinearWeights.getIndex(m, k)] -= prod;
						}
					}	
				}
			}
		}
		if(Config.displayDetail) {
			System.out.println("Gradient computation time : " + timing.stop());
		}
		return gradient;
	}
	
}
