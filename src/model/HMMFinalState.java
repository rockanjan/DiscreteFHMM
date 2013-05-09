package model;


import java.util.Random;

import corpus.Corpus;
import model.param.HMMParamFinalState;

public class HMMFinalState extends HMMBase{
	public HMMFinalState() {
		super();
		this.hmmType = HMMType.WITH_FINAL_STATE;
	}

	public HMMFinalState(int nrStates, Corpus c) {
		super();
		this.corpus = c;
		this.nrStates = nrStates;
		this.nrStatesWithFake = nrStates + 1;
		this.hmmType = HMMType.WITH_FINAL_STATE;
	}

	public void initializeRandom(Random r) {
		param = new HMMParamFinalState(this);
		param.initialize(r);
	}
	
	public void initializeZeros() {
		param = new HMMParamFinalState(this);
		param.initializeZeros();
	}
	
	public static void main(String[] args) {
		/*
		//check saving and loading model
		int nrStates = 80;
		int nrObs = 200;
		HMMFinalState hmm = new HMMFinalState(nrStates, nrObs);
		hmm.initializeRandom(new Random());
		HMMParamBase beforeSaving = new HMMParamFinalState(hmm);
		beforeSaving.initializeZeros();
		beforeSaving.cloneFrom(hmm.param);
		String fileSaved = hmm.saveModel();
		hmm.param.clear();
		hmm = null;
		
		HMMFinalState loaded = new HMMFinalState();
		loaded.loadModel(fileSaved);
		if(beforeSaving.equalsExact(loaded.param)) {
			System.out.println("Saved and Loaded models match exactly");
		} else if(beforeSaving.equalsApprox(loaded.param)) {
			System.out.println("Saved and Loaded models match approx");
		} else {
			System.out.println("Saved and Loaded models do not match");
		}
		*/
	}

	@Override
	public void initializeZerosToBest() {
		// TODO Auto-generated method stub
		
	}
}
