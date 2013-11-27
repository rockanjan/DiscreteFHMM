package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import model.param.HMMParamBase;
import model.param.HMMParamNoFinalStateLog;
import config.Config;
import corpus.Corpus;
import corpus.Vocabulary;
import corpus.WordClass;

public abstract class HMMBase {
	public int nrLayers;
	public Corpus corpus;
	public int nrStatesWithFake = -1; // the extending class should initialize
										// this (for no fake, equals nrStates)
	public int nrStates = -1;
	public HMMParamBase param;
	public HMMParamBase bestParam; // best found so far
	public int nrClasses = -1; 
	
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
		//update sufficient statistics
		for(int m=0; m<nrLayers; m++) {
			param.initial.get(m).cloneFrom(counts.initial.get(m));
			param.transition.get(m).cloneFrom(counts.transition.get(m));
		}
		//normalize
		this.param.normalize();
	}
	
	public void updateFromCountsWeighted(HMMParamBase counts, double weight) {
		//update weighted sufficient statistics
		for(int m=0; m<nrLayers; m++) {
			param.initial.get(m).cloneWeightedFrom(counts.initial.get(m), weight);
			param.transition.get(m).cloneWeightedFrom(counts.transition.get(m), weight);
		}
		//normalize
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
			pw.println(param.weightsClass.weights.length);
			pw.println(param.weightsClass.weights[0].length);
			for (int c = 0; c < param.weightsClass.weights.length; c++) {
				for (int u = 0; u < param.weightsClass.weights[0].length; u++) {
					pw.print(param.weightsClass.weights[c][u]);
					if (u != param.weightsClass.weights[0].length - 1) {
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
			int nrStatesRead = Integer.parseInt(modelReader.readLine());
			int nrLayersRead = Integer.parseInt(modelReader.readLine());
			if(nrStatesRead != nrStates) {
				System.out.format("WARNING: read number of states = %d, model = %d \n", nrStatesRead, nrStates);
				System.exit(-1);
			}
			if(nrLayersRead != nrLayers) {
				System.out.format("WARNING: read number of layers = %d, model = %d \n", nrLayersRead, nrLayers);
				System.exit(-1);
			}
			this.nrStates = nrStatesRead;
			this.nrLayers = nrLayersRead;
			
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
			for (int z = 0; z < nrLayers; z++) {
				//pw.println(param.transition.get(z).getConditionalSize());
				//pw.println(param.transition.get(z).getConditionedSize());
				modelReader.readLine();
				modelReader.readLine();
				
				for (int j = 0; j < param.transition.get(z).getConditionalSize(); j++) {
					String[] probs = modelReader.readLine().split("\\s+");
					for (int i = 0; i < param.transition.get(z).getConditionedSize(); i++) {
						param.transition.get(z).count[i][j] = Double.parseDouble(probs[i]);
					}
					//modelReader.readLine();
				}
				modelReader.readLine();
			}
			modelReader.readLine();
			// log linear weights
			//pw.println(param.weights.vocabSize);
			int vocabSizeFromModel = Integer.parseInt(modelReader.readLine());
			if(vocabSizeFromModel != Corpus.corpusVocab.get(0).vocabSize) {
				System.err.format("Vocab size mismatch, from dictionary %d, from model %d\n", Corpus.corpusVocab.get(0).vocabSize, vocabSizeFromModel);
			}
			//pw.println(param.weights.conditionalSize);
			modelReader.readLine();
			for (int y = 0; y < param.weights.vocabSize; y++) {
				String[] weights = modelReader.readLine().split("\\s+");
				for (int u = 0; u < param.weights.conditionalSize; u++) {
					param.weights.weights[y][u] = Double.parseDouble(weights[u]);
				}
				//pw.println();
			}
			int numClusterFromModel = Integer.parseInt(modelReader.readLine());
			if(numClusterFromModel != WordClass.numClusters) {
				System.err.println("numClustersFromClusterFile : " + WordClass.numClusters + " from Modelfile : " + numClusterFromModel);
			}
			modelReader.readLine();
			for (int c = 0; c < param.weightsClass.weights.length; c++) {
				String[] weightsClass = modelReader.readLine().split("\\s+");
				for (int u = 0; u < param.weightsClass.weights[0].length; u++) {
					param.weightsClass.weights[c][u] = Double.parseDouble(weightsClass[u]);
				}
			}
			if(modelReader.readLine().equals("EOF")) {
				System.out.println("Model file loaded successfully");
			} else {
				System.err.println("Model file loading failed");
				System.exit(-1);
			}
			modelReader.close();
			param.check();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
		
	}

	public void loadModelsFromIndependentHMM(String folderName) {
		//load vocab file
		File folder = new File(folderName);
		//read vocab
		if(Corpus.corpusVocab == null) {
			Corpus.corpusVocab = new ArrayList<Vocabulary>();
		}
		if(Corpus.corpusVocab.size() == 0) {
			Corpus.corpusVocab.add(new Vocabulary());
		}
		Corpus.corpusVocab.get(0).readDictionary(folder.getAbsolutePath() + "/vocab.txt");
		this.initializeZeros();
		String[] files = folder.list();
		int modelCount = 0;
		for( String modelFilename : files) {
			if(modelFilename.equals(".") || modelFilename.equals("..") || modelFilename.equals("vocab.txt")) continue;
			try {
				BufferedReader br = new BufferedReader(new FileReader(folder.getAbsolutePath() + "/" + modelFilename));
				try{
					nrStates = Integer.parseInt(br.readLine());
					nrStatesWithFake = nrStates;
					int vocabSizeFromModel = Integer.parseInt(br.readLine());
					if(vocabSizeFromModel != Corpus.corpusVocab.get(0).vocabSize) {
						System.err.println("Error: vocab size from model : " + vocabSizeFromModel + " from dictionary : " + Corpus.corpusVocab.get(0).vocabSize);
						System.exit(-1);
					}
					br.readLine();
					//load initial
					String splitted[] = br.readLine().split("(\\s+|\\t+)");
					if(nrStates != splitted.length) {
						br.close();
						throw new RuntimeException("Loading model, Initial parameters not matching number of states");
					}
					for(int i=0; i<nrStates; i++) {
						double prob = Double.parseDouble(splitted[i]);
						if(prob == 0) {
							prob = Math.log(1e-50);
						}
						else {
							prob = Math.log(prob);
						}
						this.param.initial.get(modelCount).set(i, 0, prob);
					}
					br.readLine();
					//transition
					for(int i=0; i<nrStates; i++) {
						splitted = br.readLine().split("(\\s+|\\t+)");
						if(nrStatesWithFake != splitted.length) {
							br.close();
							System.err.format("For transition: nrStates=%d, from file=%d\n", nrStatesWithFake, splitted.length);
							throw new RuntimeException("Loading model, transition parameters not matching number of states");
						}
						for(int j=0; j<splitted.length; j++) {
							double prob = Double.parseDouble(splitted[j]);
							if(prob == 0) {
								prob = Math.log(1e-50);
							}
							else {
								prob = Math.log(prob);
							}
							this.param.transition.get(modelCount).set(j, i, prob);
						}
					}
					br.readLine();
					//observation
					for(int i=0; i<nrStates; i++) {
						splitted = br.readLine().split("(\\s+|\\t+)");
						if(Corpus.corpusVocab.get(0).vocabSize != splitted.length) {
							br.close();
							System.err.format("nrStates=%d, from file=%d\n", Corpus.corpusVocab.get(0).vocabSize, splitted.length);
							throw new RuntimeException("Loading model, obs parameters not matching number of states");
						}
						for(int j=0; j<splitted.length; j++) {
							double prob = Double.parseDouble(splitted[j]);
							if(prob <= 1e-5) {
								prob = Math.log(1e-3);
							}
							else {
								prob = Math.log(prob);
							}
							this.param.weights.set(modelCount, i, j, prob);
						}
					}
					br.close();					
				} catch(NumberFormatException e) {
					e.printStackTrace();
					System.err.println("Error loading model");
					System.exit(-1);
				}			
			}
			catch (FileNotFoundException fnfe) {
				fnfe.printStackTrace();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.format("Layer %d, Model loaded successfully with %d states and %d observations \n", modelCount, nrStates, Corpus.corpusVocab.get(0).vocabSize);
			modelCount++;
		}
		param.weights.initializeUniform(0.01);
		param.check();
		System.out.println("Done");
		System.out.println("Model loaded from independent HMM with 7 layers");
	}	
	
	public static void main(String[] args) {
		Corpus corpus;
		corpus = new Corpus("\\s+", Config.vocabThreshold);
		corpus.createArtificialVocab(100);
		corpus.corpusVocab.get(0).writeDictionary(Config.baseDirModel + "vocab.txt");
		HMMBase model = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		corpus.model = model;
		model.initializeRandom(Config.random);
		model.initializeZerosToBest();
		model.param.expWeights = model.param.weights.getCloneExp();
		model.param.check();
		model.saveModel(20000);
		HMMBase loadedModel = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		loadedModel.loadModel("variational_model_layers_" + Config.nrLayers + 
					"_states_" + Config.numStates + 
					"_iter_" + 20000 + ".txt");
		if(model.param.equalsExact(loadedModel.param)) {
			System.out.println("Two models equal exactly");
		} else {
			System.err.println("Two models do not equal exactly");
		}
		//model.param.check();
	}
}
