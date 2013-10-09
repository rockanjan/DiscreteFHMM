package program;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;
import java.util.concurrent.ConcurrentHashMap;

import config.Config;

import util.Timing;
import model.HMMBase;
import model.HMMNoFinalStateLog;
import model.train.EM;
import corpus.Corpus;
import corpus.Instance;
import corpus.InstanceList;

public class Main {
	static HMMBase model;
	static Corpus corpus;
	public static void main(String[] args) throws IOException {
		corpus = new Corpus("\\s+", Config.vocabThreshold);
		trainNew();		
		//trainContinue("variational_model_layers_20_states_2_iter_21.txt");
		if(Corpus.testInstanceList != null) {
			testVariational(model, Corpus.testInstanceList, Config.outFileTest);
		} else {
			testVariational(model, Corpus.trainInstanceList, Config.outFileTrain);
		}
	}

	public static void trainNew() throws IOException {
		corpus.readVocab(Config.baseDirData + Config.vocabFile);
		Corpus.corpusVocab.get(0).writeDictionary(Config.baseDirModel + "vocab.txt");
		corpus.setupSampler();
		corpus.readTrain(Config.baseDirData + Config.trainFile);
		if(Config.testFile != null && !Config.testFile.equals("")) {
			corpus.readTest(Config.baseDirData + Config.testFile);
		}
		if(Config.devFile != null && !Config.devFile.equals("")) { 
			corpus.readDev(Config.baseDirData + Config.devFile);
		}
		model = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		corpus.model = model;
		//random init		
		model.initializeRandom(Config.random);
		model.param.weights.initializeZeros();
		//model.initializeZeros();
		model.initializeZerosToBest();
		Config.printParams();
		EM em = new EM(Config.numIter, corpus, model);
		em.start();
		model.saveModel();
	}
	
	public static void trainContinue(String filename) throws IOException {		
		corpus = new Corpus("\\s+", Config.vocabThreshold);
		model = new HMMNoFinalStateLog(Config.nrLayers, Config.numStates, corpus);
		//load model for continuing training
		model.loadModel(filename);
		//model.loadModelsFromIndependentHMM("out/model/softEM");
		corpus.model = model;
		corpus.readTrain(Config.baseDirData + Config.trainFile);
		if(Config.testFile != null && !Config.testFile.equals("")) {
			corpus.readTest(Config.baseDirData + Config.testFile);
		}
		if(Config.devFile != null && !Config.devFile.equals("")) { 
			corpus.readDev(Config.baseDirData + Config.devFile);
		}
		model.initializeZerosToBest();
		Config.printParams();
		EM em = new EM(Config.numIter, corpus, model);
		em.start();
		model.saveModel();
	}
	
	public static void testVariational(HMMBase model, InstanceList instanceList, String outFile) {
		System.out.println("Decoding variational");
		Timing decodeTiming = new Timing();
		decodeTiming.start();
		System.out.println("Decoding started on :" + new Date().toString());
		model.param.expWeights = model.param.weights.getCloneExp();
		InstanceList.featurePartitionCache = new ConcurrentHashMap<String, Double>();
		instanceList.doVariationalInference(model); //also decodes
		try{
			PrintWriter pw = new PrintWriter(Config.baseDirDecode + outFile);
			for (int n = 0; n < instanceList.size(); n++) {
				Instance i = instanceList.get(n);
				//i.decode();
				for (int t = 0; t < i.T; t++) {
					String word = i.getWord(t);
					StringBuffer sb = new StringBuffer();
					sb.append(word + " ");
					for(int m=0; m<model.nrLayers; m++) {
						int state = i.decodedStates[m][t];
						sb.append("|" + state);
					}
					pw.println(sb.toString());
					pw.flush();
				}
				pw.println();
				i.clearInference();
			}
			pw.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		model.param.expWeights = null;
		InstanceList.featurePartitionCache = null;
		System.out.println("Finished decoding");
		System.out.println("Total decoding time : " + decodeTiming.stop());
	}	
}
