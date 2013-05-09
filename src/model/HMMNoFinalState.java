package model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import corpus.Corpus;

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
		this.nrStatesWithFake = nrStates;
		this.nrStates = nrStates;
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
		// TODO Auto-generated method stub
		
	}
}
