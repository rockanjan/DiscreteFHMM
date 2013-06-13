package model.inference;

import java.util.ArrayList;

import model.HMMBase;
import model.param.HMMParamBase;
import model.param.MultinomialBase;
import util.MathUtils;
import corpus.Instance;

public class ForwardBackwardLog extends ForwardBackward{
	public ForwardBackwardLog(HMMBase model, Instance instance) {
		super();
		this.model = model;
		this.instance = instance;
		this.nrStates = model.nrStates;
		T = instance.T; 
		initial = model.param.initial.get(0);
		transition = model.param.transition.get(0);
	}
	
	@Override
	public void doInference() {
		forward();
		backward();
		computePosterior();
	}
	
	@Override
	public void forward() {
		alpha = new double[T][nrStates]; //alphas also stored in log scale
		logLikelihood = 0;
		//initialization: for t=0
		for(int i=0; i<nrStates; i++) {
			double pi = initial.get(i, 0);
			double obs = instance.getObservationProbability(0, i);
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
				obs = instance.getObservationProbability(t, j);
				alpha[t][j] = MathUtils.logsumexp(expParams) + obs; 
			}			
		}
		logLikelihood = MathUtils.logsumexp(alpha[T-1]);
		//MyArray.printExpTable(alpha, "log alpha");
		//System.out.println(logLikelihood);
	}
	
	@Override
	public void backward() {
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
					double obs = instance.getObservationProbability(t+1, j);
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
		posterior = new double[T][nrStates];
		instance.posteriors = new double[T][nrStates];
		
		for(int t=0; t<T; t++) {
			double[] expSum = new double[nrStates];
			for(int i=0; i<nrStates; i++) {
				expSum[i] = alpha[t][i] + beta[t][i];
			}
			double denom = MathUtils.logsumexp(expSum);
			for(int i=0; i<nrStates; i++) {
				//posterior[t][i] = alpha[t][i] + beta[t][i] - logLikelihood; //not working... why?
				posterior[t][i] = alpha[t][i] + beta[t][i] - denom;
				posterior[t][i] = Math.exp(posterior[t][i]);
				instance.posteriors[t][i] = posterior[t][i];
			}
		}
		//MyArray.printTable(posterior, "log posterior");
		checkStatePosterior();		
	}
	
	public void checkStatePosterior(){
		double tolerance = 1e-5;
		for(int t=0; t<T; t++) {
			double sum = 0;
			//sum = LogExp.logsumexp(posterior[t]);
			//sum = Math.exp(sum);
			for(int i=0; i<nrStates; i++) {
				//sum += Math.exp(getStatePosterior(t,i));
				sum += getStatePosterior(t,i);
			}
			if(Math.abs(sum - 1) > tolerance) {
				throw new RuntimeException("In checking state posterior, sum = " + sum);
			}
		}
	}
	
	public void addToCounts(HMMParamBase param) { 
		addToInitial(param.initial.get(0));
		addToTransition(param.transition.get(0));
	}
	
	//TODO: check if we can still work in log scale instead of exponents
	//works in normal scale (not log scale)
	public void addToInitial(MultinomialBase initial) {
		for(int i=0; i<nrStates; i++) {
			initial.addToCounts(i, 0, getStatePosterior(0, i));
		}
	}
	
	//works in normal scale (not log scale)
	public void addToObservation(ArrayList<MultinomialBase> observation) {
		for(int t=0; t<T; t++) {
			for(int i=0; i<nrStates; i++) {
				for(int k=0; k<model.corpus.oneTimeStepObsSize;k++) {
					observation.get(k).addToCounts(instance.words[t][k], i, getStatePosterior(t, i));
				}
			}
		}
	}
	
	//works in normal scale (not log scale)
	public void addToTransition(MultinomialBase transition) {
		for(int t=0; t<T-1; t++) {
			for(int i=0; i<nrStates; i++) {
				for(int j=0; j<nrStates; j++) {
					transition.addToCounts(i, j, Math.exp(getTransitionPosterior(i, j, t)));
				}
			}
		}
	}
	
	/*
	 * Gives the log of transition posterior probability (normalized by logLikelihood)
	 */
	public double getTransitionPosterior(int currentState, int nextState, int position) {
		//xi in Rabiner Tutorial
		double alphaValue = alpha[position][currentState];
		double trans = transition.get(nextState, currentState); //transition to next given current
		//double obs = model.param.observation.get(instance.words[position+1], nextState);
		double obs = instance.getObservationProbability(position+1, nextState);
		double betaValue = beta[position+1][nextState];		
		double value = alphaValue + trans + obs + betaValue - logLikelihood;		
		return value;
	}
	
	public double getStatePosterior(int t, int s) {
		return posterior[t][s];
	}
	
	@Override
	public void checkForwardBackward() {
				
	}
}
