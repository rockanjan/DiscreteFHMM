package model.inference;

import java.util.ArrayList;

import model.HMMBase;
import model.param.HMMParamBase;
import model.param.MultinomialBase;
import util.MathUtils;
import util.MyArray;
import corpus.Corpus;
import corpus.Instance;
import corpus.WordClass;

public class ForwardBackwardLog extends ForwardBackward{
	public ForwardBackwardLog(HMMBase model, Instance instance, int layer) {
		super();
		this.model = model;
		this.instance = instance;
		this.nrStates = model.states[layer];
		this.layer = layer;
		T = instance.T; 
		initial = model.param.initial.get(layer);
		transition = model.param.transition.get(layer);	
	}
	
	@Override
	public void doInference() {
		forward();
		backward();
		computePosterior(); //also computes jointObjective for states
	}
	
	
	
	@Override
	public void forward() {
		logLikelihood =0;
		//if labeled layer
		if(instance.isLabeledLayer(layer)) {
			//calculate just the loglikelihood
			//obs should be normalized for each state 
			// (to make it similar to variational step)
			double obs;
			double normalizer = 0;
			int currentWordIndex = instance.words[0][0];
			int currentClusterIndex = WordClass.wordIndexToClusterIndex.get(currentWordIndex);
			double[] obsValue = new double[model.states[layer]];
			for(int k=0; k<model.states[layer]; k++) {
				obsValue[k] = model.param.weights.get(layer, k, currentWordIndex) 
					+ model.param.weightsClass.get(layer, k, currentClusterIndex);
			}
			normalizer = MathUtils.logsumexp(obsValue);
			//normalize and update				
			for(int k=0; k<model.states[layer]; k++) {
				obsValue[k] = obsValue[k] - normalizer;				
			}
			//initial (timestep 0)
			obs = obsValue[instance.tags[0][layer]];
			double pi = initial.get(instance.tags[0][layer], 0);
			logLikelihood += pi + obs;
			//for other timesteps
			for(int t=1; t<T; t++) {
				currentWordIndex = instance.words[t][0];
				currentClusterIndex = WordClass.wordIndexToClusterIndex.get(currentWordIndex);
				for(int k=0; k<model.states[layer]; k++) {
					obsValue[k] = model.param.weights.get(layer, k, currentWordIndex) 
						+ model.param.weightsClass.get(layer, k, currentClusterIndex);
				}
				normalizer = MathUtils.logsumexp(obsValue);
				//normalize and update				
				for(int k=0; k<model.states[layer]; k++) {
					obsValue[k] = obsValue[k] - normalizer;				
				}
				obs = obsValue[instance.tags[0][layer]];
				double trans = transition.get(instance.tags[t][layer], instance.tags[t-1][layer]);
				logLikelihood += trans + obs;
			}
			return;
		}
		//reaches here only if unlabeled layer
		alpha = new double[T][nrStates]; //alphas also stored in log scale
		//initialization: for t=0
		for(int i=0; i<nrStates; i++) {
			double pi = initial.get(i, 0);
			double obs = instance.varParam.varParamObs.shi[layer][0][i];
			alpha[0][i] = pi + obs; //these prob are in logscale			
		}
		
		//induction
		for(int t = 1; t < T; t++) {
			for(int j=0; j<nrStates; j++) {
				double[] expParams = new double[nrStates];
				for(int i=0; i<nrStates; i++) {
					expParams[i] = alpha[t-1][i] + transition.get(j, i); 
				}
				double obs;
				obs = instance.varParam.varParamObs.shi[layer][t][j];
				alpha[t][j] = MathUtils.logsumexp(expParams) + obs; 
			}			
		}
		logLikelihood = MathUtils.logsumexp(alpha[T-1]);
		if(logLikelihood > 0) {
			MyArray.printExpTable(alpha, "alpha");
			throw new RuntimeException("loglikelihood is greater than zero for layer " + layer);
		}
	}
	
	@Override
	public void backward() {
		//if labeled layer, do nothing
		if(instance.isLabeledLayer(layer)) {
			return;
		}
		beta = new double[T][nrStates];
		//initialization for t=T
		for(int i=0; i<nrStates; i++) {
			beta[T-1][i] = 0.0; //log(1)
		}
		//induction
		for(int t=T-2; t>=0; t--) {
			for(int i=0; i<nrStates; i++) {			
				double[] expParams = new double[nrStates];
				for(int j=0; j<nrStates; j++) {
					double trans = transition.get(j, i);
					double obs = instance.varParam.varParamObs.shi[layer][t+1][j];
					expParams[j] = trans + obs + beta[t+1][j];
				}
				beta[t][i] = MathUtils.logsumexp(expParams);
			}
		}
		//MyArray.printExpTable(beta, "log beta");
	}
	
	//regular probablity (no log)
	@Override
	public void computePosterior() {
		double[][] oldPosterior = null;
		if(posterior != null) {
			oldPosterior = MyArray.getCloneOfMatrix(posterior);
		}
		posterior = new double[T][nrStates];
		instance.posteriors[layer] = new double[T][nrStates];
		if(instance.isLabeledLayer(layer)) {
			for(int t=0; t<T; t++) {
				//posterior[t][instance.tags[t][layer]] = 1.0;
				//smooth
				double maxParam = 0.98;
				for(int i=0; i<model.states[layer]; i++) {
					if(i == instance.tags[t][layer]) {
						posterior[t][i] = maxParam;
					} else {
						posterior[t][i] = (1-maxParam) / (model.states[layer] - 1);
					}
					instance.posteriors[layer][t][i] = posterior[t][i];
				}
				
			}
			checkStatePosterior();
			//posterior difference will be zero
			return;
			
		}
		
		for(int t=0; t<T; t++) {
			double[] expSum = new double[nrStates];
			for(int i=0; i<nrStates; i++) {
				expSum[i] = alpha[t][i] + beta[t][i];
			}
			double denom = MathUtils.logsumexp(expSum);
			for(int i=0; i<nrStates; i++) {
				posterior[t][i] = alpha[t][i] + beta[t][i] - denom;
				posterior[t][i] = Math.exp(posterior[t][i]);
				instance.posteriors[layer][t][i] = posterior[t][i];
				if(oldPosterior != null) {
					double diff = Math.abs(oldPosterior[t][i] - posterior[t][i]);
					instance.posteriorDifference += diff;
					if(diff > instance.posteriorDifferenceMax) {
						instance.posteriorDifferenceMax = diff;
					}
					
				}
			}			
		}
		checkStatePosterior();		
	}
	
	public void checkStatePosterior(){
		double tolerance = 1e-5;
		for(int t=0; t<T; t++) {
			double sum = 0;
			for(int i=0; i<nrStates; i++) {
				double value = getStatePosterior(t,i);
				if(Math.exp(value) > 1) {
					//throw new RuntimeException("State posterior prob greater than 1");
				}
				MathUtils.check(value);
				sum += value;
			}
			if(Math.abs(sum - 1) > tolerance) {
				throw new RuntimeException("In checking state posterior, sum = " + sum);
			}
		}
	}
	
	public void addToCounts(HMMParamBase param) { 
		addToInitial(param.initial.get(layer));
		addToTransition(param.transition.get(layer));
	}
	
	//works in normal scale (not log scale)
	public void addToInitial(MultinomialBase initial) {
		for(int i=0; i<nrStates; i++) {
			initial.addToCounts(i, 0, getStatePosterior(0, i));
		}
	}
	
	//works in normal scale (not log scale)
	public void addToObservation(ArrayList<MultinomialBase> observation) {
		throw new UnsupportedOperationException("not implemented");		
	}
	
	public void addToTransition(MultinomialBase transition) {
		if(instance.isLabeledLayer(layer)) {
			for(int t=0; t<T-1; t++) {
				transition.addToCounts(instance.tags[t+1][layer], instance.tags[t][layer], 1.0);
			}
			return;
		}
		for(int t=0; t<T-1; t++) {
			for(int i=0; i<nrStates; i++) {
				for(int j=0; j<nrStates; j++) {
					//opposite indices because finally we want j given i (using prob i->j)
					transition.addToCounts(j, i, getTransitionPosterior(i, j, t));
				}
			}
		}
	}
	
	/*
	 * Gives the numerator of transition posterior probability
	 * P(S_t-1, S_t | O)
	 */
	public double getTransitionPosterior(int currentState, int nextState, int position) {
		if(instance.isLabeledLayer(layer)) {
			if(instance.tags[position][layer] == currentState &&
					instance.tags[position+1][layer] == nextState) {
				return 1.0;
			}
			return 0;
		}
		//xi in Rabiner Tutorial
		double alphaValue = alpha[position][currentState];
		double trans = transition.get(nextState, currentState); //transition to next given current
		double obs = instance.varParam.varParamObs.shi[layer][position+1][nextState];		
		double betaValue = beta[position+1][nextState];		
		double value = alphaValue + trans + obs + betaValue - logLikelihood;
		value = Math.exp(value);
		if(value > 1 + 1e-2) {
			throw new RuntimeException("transition posterior value : " + value);
		}
		return value;
	}
	
	public double getStatePosterior(int t, int s) {
		return posterior[t][s];
	}
	
	@Override
	public void checkForwardBackward() {
				
	}
}
