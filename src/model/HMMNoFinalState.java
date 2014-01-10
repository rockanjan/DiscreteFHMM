package model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import corpus.Corpus;
import corpus.Instance;

import model.param.HMMParamBase;
import model.param.HMMParamFinalState;
import model.param.HMMParamNoFinalState;

public class HMMNoFinalState extends HMMBase{
	public HMMNoFinalState() {
		super();
		this.hmmType = HMMType.WITH_NO_FINAL_STATE;
	}

	public HMMNoFinalState(int nrStates, Corpus corpus) {
		super();
		this.corpus = corpus;
		this.hmmType = HMMType.WITH_NO_FINAL_STATE;
	}

	public void initializeRandom(Random r) {
		this.param = new HMMParamNoFinalState(this);
		this.param.initialize(r);
	}
	
	public void initializeZeros() {
		param = new HMMParamNoFinalState(this);
		param.initializeZeros();
	}
	
	public static void main(String[] args) {
		/*
		//check saving and loading model
		int nrStates = 20;
		int nrObs = 50;
		HMMNoFinalState hmm = new HMMNoFinalState(nrStates, nrObs);
		hmm.initializeRandom(new Random());
		HMMParamBase beforeSaving = new HMMParamNoFinalState(hmm);
		beforeSaving.initializeZeros();
		beforeSaving.cloneFrom(hmm.param);
		String fileSaved = hmm.saveModel();
		hmm.param.clear();
		hmm = null;
		
		HMMNoFinalState loaded = new HMMNoFinalState();
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
		bestParam = new HMMParamNoFinalState(this);
		bestParam.initializeZeros();
	}
	
	
	@Override
	public void computePreviousTransitions() {
		//for all the training corpus
		for(int i=0; i<corpus.trainInstanceList.size(); i++) {
			Instance sentence = corpus.trainInstanceList.get(i);
			//initial
			for(int z=1; z<corpus.oneTimeStepObsSize; z++) {
				double prevValue = this.param.initial.get(z).get( sentence.words[0][z], 0);
				this.param.initial.get(z).addToCounts( sentence.words[0][z], 0, 1);
			}
			
			for(int t=1; t<sentence.T; t++) {
				for(int z=1; z<corpus.oneTimeStepObsSize; z++) {
					double prevValue = this.param.transition.get(z).get( sentence.words[t][z], sentence.words[t-1][z]);
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
