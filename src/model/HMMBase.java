package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import config.Config;

import corpus.Corpus;
import corpus.Vocabulary;

import model.param.HMMParamBase;
import model.param.HMMParamNoFinalState;
import model.param.HMMParamNoFinalStateLog;
import model.param.LogLinearWeights;
import model.param.MultinomialBase;
import model.param.MultinomialLog;

public abstract class HMMBase {
	public int nrLayers;
	public Corpus corpus;
	public int nrStatesWithFake = -1; // the extending class should initialize
										// this (for no fake, equals nrStates)
	public int nrStates = -1;
	public HMMParamBase param;
	public HMMParamBase bestParam; // best found so far

	public HMMType hmmType;
	
	public abstract void initializeRandom(Random r);

	public abstract void initializeZeros();

	public abstract void initializeZerosToBest();

	/*
	 * computes the transition probabilities for the states decoded in the
	 * previous layer (Z variables)
	 */
	public abstract void computePreviousTransitions();

	public void checkModel() {
		param.check();
	}

	public void updateFromCounts(HMMParamBase counts) {
		counts.normalize();
		for(int m=0; m<nrLayers; m++) {
			param.initial.get(m).cloneFrom(counts.initial.get(m));
			param.transition.get(m).cloneFrom(counts.transition.get(m));
		}
	}
	
	public void updateFromCountsWeighted(HMMParamBase counts, double weight) {
		for(int m=0; m<nrLayers; m++) {
			param.initial.get(m).cloneWeightedFrom(counts.initial.get(m), weight);
			param.initial.get(m).normalize();
			param.transition.get(m).cloneWeightedFrom(counts.transition.get(m), weight);
			param.transition.get(m).normalize();
		}
		//renormalize
		this.param.normalize();
	}

	public String saveModel() {
		return saveModel(-1);
	}

	/*
	 * return the location saved
	 */
	public String saveModel(int iterCount) {
		File folder = new File(Config.baseDirModel);
		if (!folder.exists()) {
			folder.mkdir();
		}
		String modelFile = "";
		if (iterCount < 0) {
			modelFile = folder.getAbsolutePath() + "/variational_model_layers_" + nrLayers +  "_states_" + nrStates + "_final.txt";
		} else {
			modelFile = folder.getAbsolutePath() + "/variational_model_layers_" + nrLayers + "_states_" + nrStates + "_iter_" + iterCount + ".txt";
		}
		PrintWriter pw;
		try {
			pw = new PrintWriter(modelFile);
			pw.println(nrStates);
			pw.println(nrLayers);
			pw.println();
			// initial
			for (int z = 0; z < nrLayers; z++) {
				pw.println(param.initial.get(z).getConditionedSize());
				for (int i = 0; i < param.initial.get(z).getConditionedSize(); i++) {
					pw.print(param.initial.get(z).get(i, 0));
					if (i != param.initial.get(z).getConditionedSize() - 1) {
						pw.print(" ");
					}
				}
				pw.println();
			}
			pw.println();
			// transition
			for (int z = 0; z < nrLayers; z++) {
				pw.println(param.transition.get(z).getConditionalSize());
				pw.println(param.transition.get(z).getConditionedSize());
				for (int j = 0; j < param.transition.get(z)
						.getConditionalSize(); j++) {
					for (int i = 0; i < param.transition.get(z)
							.getConditionedSize(); i++) {
						pw.print(param.transition.get(z).get(i, j));
						if (i != param.transition.get(z).getConditionedSize() - 1) {
							pw.print(" ");
						}
					}
					pw.println();
				}
				pw.println();
			}
			pw.println();
			// log linear weights
			pw.println(param.weights.vocabSize);
			pw.println(param.weights.conditionalSize);
			for (int y = 0; y < param.weights.vocabSize; y++) {
				for (int u = 0; u < param.weights.conditionalSize; u++) {
					pw.print(param.weights.weights[y][u]);
					if (u != param.weights.conditionalSize - 1) {
						pw.print(" ");
					}
				}
				pw.println();
			}
			pw.println("EOF");
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		return folder.getAbsolutePath();
	}

	public void loadModel(String filename) {
		//load vocab file
		File folder = new File(Config.baseDirModel);
		if(filename == null || filename.equals("")) {
			filename = "variational_model_layers_" + nrLayers +  "_states_" + nrStates + "_final.txt";
		}
		String modelFile = folder.getAbsolutePath() + "/" + filename;
		if(Corpus.corpusVocab == null) {
			Corpus.corpusVocab = new ArrayList<Vocabulary>();
		}
		if(Corpus.corpusVocab.size() == 0) {
			Corpus.corpusVocab.add(new Vocabulary());
		}
		Corpus.corpusVocab.get(0).readDictionary(folder.getAbsolutePath() + "/vocab.txt");
		BufferedReader modelReader;
		try {
			modelReader = new BufferedReader(new FileReader(modelFile));
			this.nrStates = Integer.parseInt(modelReader.readLine());
			this.nrLayers = Integer.parseInt(modelReader.readLine());
			this.nrStatesWithFake = this.nrStates;
			
			modelReader.readLine(); //empty line
			
			param = new HMMParamNoFinalStateLog(this);
			param.initializeZeros();
			param.nrObs = Corpus.corpusVocab.get(0).vocabSize;
			// initial
			
			for (int z = 0; z < nrLayers; z++) {
				//pw.println(param.initial.get(z).getConditionedSize());
				modelReader.readLine();
				String[] initialProbs = modelReader.readLine().split("\\s+");
				for (int i = 0; i < param.initial.get(z).getConditionedSize(); i++) {
					param.initial.get(z).count[i][0] = Double.parseDouble(initialProbs[i]);
				}
				//modelReader.readLine();
			}
			modelReader.readLine();
			// transition
			for (int z = 0; z < nrLayers; z++) {
				//pw.println(param.transition.get(z).getConditionalSize());
				//pw.println(param.transition.get(z).getConditionedSize());
				modelReader.readLine();
				modelReader.readLine();
				
				for (int j = 0; j < param.transition.get(z).getConditionalSize(); j++) {
					String[] probs = modelReader.readLine().split("\\s+");
					for (int i = 0; i < param.transition.get(z).getConditionedSize(); i++) {
						param.transition.get(z).count[j][i] = Double.parseDouble(probs[i]);
					}
					//modelReader.readLine();
				}
				modelReader.readLine();
			}
			modelReader.readLine();
			// log linear weights
			//pw.println(param.weights.vocabSize);
			modelReader.readLine();
			//pw.println(param.weights.conditionalSize);
			modelReader.readLine();
			for (int y = 0; y < param.weights.vocabSize; y++) {
				String[] weights = modelReader.readLine().split("\\s+");
				for (int u = 0; u < param.weights.conditionalSize; u++) {
					param.weights.weights[y][u] = Double.parseDouble(weights[u]);
				}
				//pw.println();
			}
			//pw.println("EOF");
			if(modelReader.readLine().equals("EOF")) {
				System.out.println("Model file loaded successfully");
			} else {
				System.err.println("Model file loading failed");
			}
			modelReader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
