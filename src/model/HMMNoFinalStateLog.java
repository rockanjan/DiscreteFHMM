package model;

import java.util.Random;

import corpus.Corpus;
import corpus.Instance;
import model.param.HMMParamNoFinalState;
import model.param.HMMParamNoFinalStateLog;

public class HMMNoFinalStateLog extends HMMBase{
	public HMMNoFinalStateLog() {
		super();
		this.hmmType = HMMType.LOG_SCALE;
	}

	public HMMNoFinalStateLog(int nrLayers, int nrStates, Corpus c) {
		super();
		this.nrStatesWithFake = nrStates;
		this.nrStates = nrStates;
		this.corpus = c;
		this.hmmType = HMMType.LOG_SCALE;
		this.nrLayers = nrLayers; 
	}

	public void initializeRandom(Random r) {
		this.param = new HMMParamNoFinalStateLog(this);
		this.param.initialize(r);
	}
	
	public void initializeZeros() {
		param = new HMMParamNoFinalState(this);
		param.initializeZeros();
	}
	
	@Override
	public void initializeZerosToBest() {
		bestParam = new HMMParamNoFinalState(this);
		bestParam.initializeZeros();
	}
	
	
	@Override
	public void computePreviousTransitions() {
		//for all the training corpus
		for(int i=0; i<Corpus.trainInstanceList.size(); i++) {
			Instance sentence = Corpus.trainInstanceList.get(i);
			//initial
			for(int z=1; z<Corpus.oneTimeStepObsSize; z++) {
				this.param.initial.get(z).addToCounts( sentence.words[0][z], 0, 1);
			}
			
			for(int t=1; t<sentence.T; t++) {
				for(int z=1; z<Corpus.oneTimeStepObsSize; z++) {
					this.param.transition.get(z).addToCounts( sentence.words[t][z], sentence.words[t-1][z] , 1);
				}
			}
		}
		
		for(int z=1; z<corpus.oneTimeStepObsSize; z++) {
			this.param.initial.get(z).normalize();
			this.param.initial.get(z).checkDistribution();
			this.param.transition.get(z).normalize();
			this.param.transition.get(z).checkDistribution();			
		}
	}
	
	
}
