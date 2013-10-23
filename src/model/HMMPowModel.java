package model;

import java.util.ArrayList;
import java.util.List;

import model.param.MultinomialBase;
import model.param.MultinomialLog;
import util.MathUtils;
import util.MyArray;
import config.Config;

public class HMMPowModel {
	public int powStateSize; // K^M
	public MultinomialBase initial;
	public MultinomialBase transition;
	public MultinomialBase observation;
	public static List<StateCombination> stateCombinationList;
	HMMBase model;
	public HMMPowModel(HMMBase model) {
		this.model = model;
		init();		
	}
	
	//create power model
	private void init() {
		powStateSize = (int) Math.pow(model.nrStates, model.nrLayers);
		populateStateCombination();
		initial = new MultinomialLog(powStateSize, 1);
		transition = new MultinomialLog(powStateSize, powStateSize);
		observation = new MultinomialLog(model.param.weights.vocabSize, powStateSize);
		//initialize based on the model
		for(int i=0; i<powStateSize; i++) {
			//initial
			double initProbLog = 0.0;
			int[] stateCombination = getStateCombination(i).states;
			for(int m=0; m<model.nrLayers; m++) {
				initProbLog += model.param.initial.get(m).get(stateCombination[m], 0);				
			}
			initial.set(i, 0, initProbLog);
			
			//transition
			for(int j=0; j<powStateSize; j++) {
				double transProbLog = 0.0;
				int[] nextStateCombination = getStateCombination(j).states;
				for(int m=0; m<model.nrLayers; m++) {
					transProbLog += model.param.transition.get(m).get(nextStateCombination[m], stateCombination[m]);
				}
				transition.set(j, i, transProbLog);
			}
			
			//observation (expensive)
			double[] numeratorLog = new double[model.param.weights.vocabSize];
			for(int v=0; v<model.param.weights.vocabSize; v++) {
				double obsProbLog = 0.0;
				for(int m=0; m<model.nrLayers; m++) {
					obsProbLog += model.param.weights.get(m, stateCombination[m], v);
				}
				numeratorLog[v] = obsProbLog;
			}
			//normalize
			double normalizer = MathUtils.logsumexp(numeratorLog);
			for(int v=0; v<model.param.weights.vocabSize; v++) {
				numeratorLog[v] -= normalizer;
				observation.set(v, i, numeratorLog[v]);
			}
		}
		/*
		//should not be necessary, but workaround for now
		transition.count = MathUtils.expArray(transition.count);
		transition.normalize();
		*/
		
		//checkdistribution
		initial.checkDistribution();
		//transition.checkDistribution();
		observation.checkDistribution();
	}
	
	private void populateStateCombination() {
		stateCombinationList = new ArrayList<StateCombination>();
		for(int s=0; s<powStateSize; s++) {
			stateCombinationList.add(getStateCombination(s));
		}
		//debugStateCombination();
	}
	
	public static StateCombination getStateCombination(int index) {
		int[] stateCombinationVector = new int[Config.nrLayers];
		int stateIndex = 0;
		for(int pow=Config.nrLayers-1; pow>=0; pow--) {
			int divisor = (int) Math.pow(Config.numStates, pow);
			int state = index / divisor; //quotient
			stateCombinationVector[stateIndex++] = state;
			index = index - divisor * state; //remainder 
		}
		StateCombination stateCombination = new StateCombination(stateCombinationVector);
		return stateCombination;
	}
	
	public static void debugStateCombination() {
		for(int i=0; i<stateCombinationList.size(); i++) {
			System.out.println(i + " --> " + stateCombinationList.get(i));
		}
	}
}
