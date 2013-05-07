package model.param;

import java.util.ArrayList;
import java.util.Random;

import model.HMMBase;
import model.HMMType;

public abstract class HMMParamBase {
	public MultinomialBase initial;
	public MultinomialBase transition;
	public ArrayList<MultinomialBase> observation;
	
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
			initial = new MultinomialLog(nrStates, 1);
			transition = new MultinomialLog(nrStatesWithFake, nrStates);
			observation = new ArrayList<MultinomialBase>();
			for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
				MultinomialLog temp = new MultinomialLog(model.corpus.corpusVocab.get(i).vocabSize, nrStates);
				observation.add(temp);
			}
		} else {
			initial = new MultinomialRegular(nrStates, 1);
			transition = new MultinomialRegular(nrStatesWithFake, nrStates);
			observation = new ArrayList<MultinomialBase>();
			for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
				MultinomialRegular temp = new MultinomialRegular(model.corpus.corpusVocab.get(i).vocabSize, nrStates);
				observation.add(temp);
			}
		}
	}
	
	public void initialize(Random r) {
		if(model.hmmType == HMMType.LOG_SCALE) {
			initial = new MultinomialLog(nrStates, 1);
			transition = new MultinomialLog(nrStatesWithFake, nrStates);
			observation = new ArrayList<MultinomialBase>();
			for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
				MultinomialLog temp = new MultinomialLog(model.corpus.corpusVocab.get(i).vocabSize, nrStates);
				observation.add(temp);
			}
		} else {
			initial = new MultinomialRegular(nrStates, 1);
			transition = new MultinomialRegular(nrStatesWithFake, nrStates);
			observation = new ArrayList<MultinomialBase>();
			for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
				MultinomialRegular temp = new MultinomialRegular(model.corpus.corpusVocab.get(i).vocabSize, nrStates);
				observation.add(temp);
			}
		}
		initial.initializeRandom(r);
		transition.initializeRandom(r);
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			observation.get(i).initializeRandom(r);
		}
		
	}
	
	public void check() { 
		initial.checkDistribution();
		transition.checkDistribution();
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			observation.get(i).checkDistribution();
		}
	}
	
	public void normalize() {
		initial.normalize();
		transition.normalize();
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			observation.get(i).normalize();
		}
	}
	
	public void cloneFrom(HMMParamBase source) {
		initial.cloneFrom(source.initial);
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			observation.get(i).cloneFrom(source.observation.get(i));
		}
		transition.cloneFrom(source.transition);
	}
	
	public void clear() {
		initial = null;
		transition = null;
		observation = null;
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
		
		if(! this.initial.equalsExact(other.initial) && this.transition.equalsExact(other.transition)) {
			result = false;
		}
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			if(! this.observation.get(i).equalsExact(other.observation.get(i))) {
				result = false;
			}
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
		
		if(! this.initial.equalsApprox(other.initial) && this.transition.equalsApprox(other.transition)) {
			result = false;
		}
		for(int i=0; i<model.corpus.oneTimeStepObsSize; i++) {
			if(! this.observation.get(i).equalsApprox(other.observation.get(i))) {
				result = false;
			}
		}
		return result;
	}
}
