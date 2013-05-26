package model.param;

import java.util.ArrayList;
import java.util.Random;

import model.HMMBase;
import model.HMMType;

public abstract class HMMParamBase {
	public ArrayList<MultinomialBase> initial;
	public ArrayList<MultinomialBase> transition;
	
	public LogLinearWeights weights;
	
	public HMMBase model;

	int nrStatesWithFake = -1; //the extending class should initialize this (for no fake, equals nrStates)
	int nrStates = -1;
	int nrObs = -1;
	
	
	public HMMParamBase(HMMBase model) {
		this.model = model;
		nrStates = model.nrStates;
	}
	
	public void initializeZeros() {
		if(model.hmmType == HMMType.LOG_SCALE) {
			initial = new ArrayList<MultinomialBase>();
			transition = new ArrayList<MultinomialBase>();
			
			MultinomialLog initialCurrent = new MultinomialLog(nrStates, 1);
			MultinomialLog transitionCurrent = new MultinomialLog(nrStates, nrStates); //for current hidden states
			
			initial.add(initialCurrent);
			transition.add(transitionCurrent);
			for(int i=1; i<model.corpus.oneTimeStepObsSize; i++) {
				MultinomialLog tempTrans = new MultinomialLog(model.corpus.corpusVocab.get(i).vocabSize, model.corpus.corpusVocab.get(i).vocabSize);
				MultinomialLog tempInit = new MultinomialLog(model.corpus.corpusVocab.get(i).vocabSize, 1);
				initial.add(tempInit);
				transition.add(tempTrans);
			}
			//initial random weights
			int zSize = 0;
			for(int i=1; i<model.corpus.oneTimeStepObsSize; i++) {
				zSize += model.corpus.corpusVocab.get(i).vocabSize;
			}
			//initialize weights for the log-linear model
			weights = new LogLinearWeights(model.corpus.corpusVocab.get(0).vocabSize, nrStates + zSize);
			weights.initializeZeros();
		}
	}
	
	public void initialize(Random r) {
		initializeZeros();
		initial.get(0).initializeRandom(r);
		transition.get(0).initializeRandom(r);
		//others still initialized to zero
		
		//initial random weights
		int zSize = 0;
		for(int i=1; i<model.corpus.oneTimeStepObsSize; i++) {
			initial.get(i).initializeUniformCounts();
			transition.get(i).initializeUniformCounts();
			zSize += model.corpus.corpusVocab.get(i).vocabSize;			
		}
		weights = new LogLinearWeights(model.corpus.corpusVocab.get(0).vocabSize, nrStates + zSize);
		weights.initializeRandom(r);
	}
	
	public void check() { 
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			initial.get(i).checkDistribution();
			transition.get(i).checkDistribution();
		}
	}
	
	public void normalize() {
		initial.get(0).normalize();
		transition.get(0).normalize();		
	}
	
	public void cloneFrom(HMMParamBase source) {
		
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			initial.get(i).cloneFrom(source.initial.get(i));
			transition.get(i).cloneFrom(source.transition.get(i));
		}
		
		this.weights = source.weights;
	}
	
	public void clear() {
		initial = null;
		transition = null;		
	}
	
	@Override
	public boolean equals(Object other) {
		System.err.println("NOT IMPLEMENTED");
		return false;
	}
	
	public boolean equalsExact(HMMParamBase other) {
		boolean result = true;
		if(nrStates != other.nrStates || nrObs != other.nrObs || nrStatesWithFake != other.nrStatesWithFake) {
			return false;
		}
		
		if(! this.initial.get(0).equalsExact(other.initial.get(0)) && this.transition.get(0).equalsExact(other.transition.get(0))) {
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
		return result;
	}
}
