package model.inference;

import java.util.ArrayList;

import model.HMMBase;
import model.param.HMMParamBase;
import model.param.MultinomialBase;
import util.MathUtils;
import util.MyArray;
import corpus.Instance;

public class ForwardBackwardLog extends ForwardBackward{
	public ForwardBackwardLog(HMMBase model, Instance instance, int layer) {
		super();
		this.model = model;
		this.instance = instance;
		this.nrStates = model.nrStates;
		this.layer = layer;
		T = instance.T; 
		initial = model.param.initial.get(layer);
		transition = model.param.transition.get(layer);	
	}
	
	@Override
	public void doInference() {
		forward();
		backward();
		computePosterior();
	}
	
	@Override
	public void forward() {
		logLikelihood =0;
		alpha = new double[T][nrStates]; //alphas also stored in log scale
		//initialization: for t=0
		for(int i=0; i<nrStates; i++) {
			double pi = initial.get(i, 0);
			if(pi == 0) {
				System.out.println("initial prob one");
			}
			double obs = instance.varParamObs.shi[layer][0][i];
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
				obs = instance.varParamObs.shi[layer][t][j];
				alpha[t][j] = MathUtils.logsumexp(expParams) + obs; 
			}			
		}
		logLikelihood = MathUtils.logsumexp(alpha[T-1]);
		if(logLikelihood >= 0) {
			MyArray.printExpTable(alpha, "alpha");
			throw new RuntimeException("loglikelihood is greater or equal to zero for layer " + layer);
		}
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
					double obs = instance.varParamObs.shi[layer][t+1][j];
					if(obs > 1) {
						System.err.println("Obs prob greater than 1");
					}
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
		instance.posteriors[layer] = new double[T][nrStates];
		
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
			}
		}
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
		addToInitial(param.initial.get(layer));
		addToTransition(param.transition.get(layer));
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
		throw new UnsupportedOperationException("not implemented");		
	}
	
	//works in normal scale (not log scale)
	public void addToTransition(MultinomialBase transition) {
		for(int t=0; t<T-1; t++) {
			for(int i=0; i<nrStates; i++) {
				for(int j=0; j<nrStates; j++) {
					transition.addToCounts(j, i, Math.exp(getTransitionPosterior(i, j, t)));
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
		double obs = instance.varParamObs.shi[layer][position+1][nextState];		
		double betaValue = beta[position+1][nextState];		
		//WARNING: important
		//TODO: decide if we should subtract the log likelihood
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
