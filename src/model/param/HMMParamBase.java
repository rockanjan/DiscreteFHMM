package model.param;

import java.util.ArrayList;
import java.util.Random;

import corpus.Corpus;

import util.MyArray;

import model.HMMBase;
import model.HMMType;

public abstract class HMMParamBase {
	public int nrLayers;
	public ArrayList<MultinomialBase> initial;
	public ArrayList<MultinomialBase> transition;
	
	public LogLinearWeights weights;
	
	public double[][] expWeightsCache;
	
	public HMMBase model;

	int nrStatesWithFake = -1; //the extending class should initialize this (for no fake, equals nrStates)
	int nrStates = -1;
	int nrObs = -1;
	
	
	public HMMParamBase(HMMBase model) {
		this.model = model;
		nrStates = model.nrStates;
		this.nrLayers = model.nrLayers;
	}
	
	public void initializeZeros() {
		if(model.hmmType == HMMType.LOG_SCALE) {
			initial = new ArrayList<MultinomialBase>();
			transition = new ArrayList<MultinomialBase>();
			for(int i=0; i<nrLayers; i++) {
				MultinomialLog tempTrans = new MultinomialLog(nrStates, nrStates);
				MultinomialLog tempInit = new MultinomialLog(nrStates, 1);
				initial.add(tempInit);
				transition.add(tempTrans);
			}
			//initialize weights for the log-linear model
			weights = new LogLinearWeights(Corpus.corpusVocab.get(0).vocabSize, nrLayers * nrStates);
			weights.initializeZeros();
		} else {
			throw new UnsupportedOperationException("Not implemented");
		}
	}
	
	public void initialize(Random r) {
		initializeZeros();
		for(int i=0; i<nrLayers; i++) {
			initial.get(i).initializeRandom(r);
			transition.get(i).initializeRandom(r);			
		}
		weights = new LogLinearWeights(Corpus.corpusVocab.get(0).vocabSize, nrLayers * nrStates);
		weights.initializeRandom(r);
		//TODO: normalize observation weights
	}
	
	public void initializeWeightsFromPreviousRecursion(double[][] prev) {
		throw new UnsupportedOperationException("should not be used in variational case");
		/*
		if(prev.length != weights.weights.length) {
			throw new RuntimeException("Weight initialization from previous recursion found different vocab dimension, prev = " 
					+ prev.length + " new = " + weights.weights.length);
		}
		if(prev[0].length + nrStates != weights.weights[0].length) {
			throw new RuntimeException("Weight initialization from previous recursion conditional dimension mismatch, prev = "
					+ prev[0].length + " new = " + weights.weights[0].length + " nrStates = " + nrStates );
		}
		for(int v=0; v<prev.length; v++) {
			int index = 0;
			for(int c = weights.weights[0].length; c<weights.weights[0].length; c++) {
				if(c >= 1 && c<=nrStates) {
					continue;
				}
				weights.weights[v][c] = prev[v][index];
				index++;
			}
		}
		*/
	}
	
	public void check() { 
		for(int i=0; i<nrLayers; i++) {
			initial.get(i).checkDistribution();
			transition.get(i).checkDistribution();
		}
	}
	
	public void normalize() {
		for(int i=0; i<nrLayers; i++) {
			initial.get(i).normalize();
			transition.get(i).normalize();
		}		
	}
	
	public void cloneFrom(HMMParamBase source) {
		for(int i=0; i<nrLayers; i++) {
			initial.get(i).cloneFrom(source.initial.get(i));
			transition.get(i).cloneFrom(source.transition.get(i));
		}
		this.weights.weights = MyArray.getCloneOfMatrix(source.weights.weights);
	}
	
	public void clear() {
		initial = null;
		transition = null;		
	}
	
	@Override
	public boolean equals(Object other) {
		throw new UnsupportedOperationException("Not yet implemented");
	}
	
	public boolean equalsExact(HMMParamBase other) {
		boolean result = true;
		if(nrStates != other.nrStates || nrObs != other.nrObs || nrStatesWithFake != other.nrStatesWithFake) {
			return false;
		}
		
		if(! this.initial.get(0).equalsExact(other.initial.get(0)) && this.transition.get(0).equalsExact(other.transition.get(0))) {
			result = false;
		}
		if(! weights.equalsExact(other.weights)) {
			result = false;
		}
		return result;
	}
	
	public boolean equalsApprox(HMMParamBase other) {
		if(nrStates != other.nrStates || nrObs != other.nrObs || nrStatesWithFake != other.nrStatesWithFake) {
			return false;
		}
		boolean result = true;
		if(nrStates != other.nrStates || nrObs != other.nrObs || nrStatesWithFake != other.nrStatesWithFake) {
			return false;
		}
		
		if(! this.initial.get(0).equalsApprox(other.initial.get(0)) && this.transition.get(0).equalsApprox(other.transition.get(0))) {
			result = false;
		}
		
		if(! weights.equalsApprox(other.weights)) {
			result = false;
		}
		return result;
	}
}
