package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

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
	public String baseDir = "out/model/";

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
		// don't try to update the initial and transition for the previously
		// learnt layers (z1, z2...)
		// update only for the current hidden layer x
		param.initial.get(0).cloneFrom(counts.initial.get(0));
		param.transition.get(0).cloneFrom(counts.transition.get(0));
	}

	public String saveModel() {
		return saveModel(-1);
	}

	/*
	 * return the location saved
	 */
	public String saveModel(int iterCount) {
		File folder = new File(baseDir);
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

	public void loadModel(int recursionLevel) {
		System.out.println("RECURSION : " + recursionLevel);
		System.out.println("---------");
		File folder = new File(baseDir + "recursion_" + recursionLevel);
		if (!folder.exists()) {
			throw new RuntimeException(
					"The recursion folder does not exist for loading model, recursionLevel = "
							+ recursionLevel);
		}
		String modelFile = "";
		modelFile = folder.getAbsolutePath() + "/model_states_" + nrStates + "_final.txt";
		try {
			BufferedReader modelReader = new BufferedReader(new FileReader(modelFile));
			this.nrStates = Integer.parseInt(modelReader.readLine());
			this.nrStatesWithFake = this.nrStates;
			Corpus.oneTimeStepObsSize = Integer
					.parseInt(modelReader.readLine());
			// read blank
			modelReader.readLine();

			// initialize corpus vocabs
			corpus.corpusVocab = new ArrayList<Vocabulary>();
			for (int z = 0; z < Corpus.oneTimeStepObsSize; z++) {
				Vocabulary tempVocab = new Vocabulary();
				corpus.corpusVocab.add(tempVocab);
			}
			// load vocab files
			for (int z = 0; z < Corpus.oneTimeStepObsSize; z++) {
				String vocabFilename = folder.getAbsoluteFile() + "/vocab_" + z + ".txt";
				corpus.corpusVocab.get(z).readDictionary(vocabFilename);
				//corpus.corpusVocab.get(z).debug();
			}
			
			// initialize params
			this.param = new HMMParamNoFinalStateLog(this);
			this.param.initial = new ArrayList<MultinomialBase>();
			this.param.transition = new ArrayList<MultinomialBase>();
			// read initial parameters
			for (int z = 0; z < Corpus.oneTimeStepObsSize; z++) {
				int conditionedSize = Integer.parseInt(modelReader.readLine());
				MultinomialBase init = new MultinomialLog(conditionedSize, 1);
				this.param.initial.add(init);
				String[] splittedParams = modelReader.readLine().split("\\s+");
				for (int i = 0; i < param.initial.get(z).getConditionedSize(); i++) {
					param.initial.get(z).set(i, 0, Double.parseDouble(splittedParams[i]));
				}				
			}
			modelReader.readLine();
			//read transition params
			for (int z = 0; z < Corpus.oneTimeStepObsSize; z++) {
				int conditionalSize = Integer.parseInt(modelReader.readLine());
				int conditionedSize = Integer.parseInt(modelReader.readLine());
				
				MultinomialBase transition = new MultinomialLog(conditionedSize, conditionalSize);
				this.param.transition.add(transition);
				for(int j=0; j<conditionalSize; j++) {
					String[] splittedParams = modelReader.readLine().split("\\s+");
					for (int i = 0; i < conditionedSize; i++) {
						param.transition.get(z).set(i, j, Double.parseDouble(splittedParams[i]));
					}
				}
				modelReader.readLine(); //read a blank line				
			}
			modelReader.readLine();
			// read log linear weights
			int vocabSize = Integer.parseInt(modelReader.readLine());
			int conditionalSize = Integer.parseInt(modelReader.readLine());
			
			//initialize log linear weights
			param.weights = new LogLinearWeights(vocabSize, conditionalSize-1); //-1 because initialization will add one
			param.weights.initializeZeros();
			for (int y = 0; y < param.weights.vocabSize; y++) {
				String[] splittedParams = modelReader.readLine().split("\\s+");
				for (int u = 0; u < param.weights.conditionalSize; u++) {
					param.weights.weights[y][u] = Double.parseDouble(splittedParams[u]);
				}
			}
			//verify complete
			String line = modelReader.readLine().trim();
			modelReader.close();
			if(! line.equals("EOF")) {
				throw new RuntimeException("EOF not encountered after loading model, found :" + line);				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} // read blank line
	}
}
