package corpus;

import java.util.ArrayList;

import util.SmoothWord;

import model.HMMBase;
import model.HMMType;
import model.inference.ForwardBackward;
import model.inference.ForwardBackwardLog;

public class Instance {
	public int[][] words;
	public int T; // sentence length
	Corpus c;
	public ForwardBackward forwardBackward;
	public int nrStates;
	public int unknownCount;

	public double[][] observationCache;

	public Instance(Corpus c, String line) {
		this.c = c;
		unknownCount = 0;
		// read from line
		populateWordArray(line);
	}

	public void doInference(HMMBase model) {
		// forwardBackward = new ForwardBackwardNoScaling(model, this);
		if (model.hmmType == HMMType.LOG_SCALE) {
			forwardBackward = new ForwardBackwardLog(model, this);
		} else {
			System.out.println("ONLY LOG FORWARD BACKWARD IMPLEMENTED");
			System.exit(-1);
			//forwardBackward = new ForwardBackwardScaled(model, this);
			// forwardBackward = new ForwardBackwardNoScaling(model, this);
		}
		nrStates = forwardBackward.model.nrStates;
		forwardBackward.doInference();
	}

	public void clearInference() {
		forwardBackward.clear();
		forwardBackward = null;
		observationCache = null;
	}

	public double getObservationProbability(int position, int state) {
		if (observationCache == null) {
			observationCache = new double[T][nrStates];
			for (int t = 0; t < T; t++) {
				for (int i = 0; i < nrStates; i++) {
					if (forwardBackward.model.hmmType == HMMType.LOG_SCALE) {
						observationCache[t][i] = 0.0;
						for (int k = 0; k < c.oneTimeStepObsSize; k++) {
							observationCache[t][i] += forwardBackward.model.param.observation
									.get(k).get(words[t][k], i);
						}
					} else {
						observationCache[t][i] = 1.0;
						for (int k = 0; k < c.oneTimeStepObsSize; k++) {
							observationCache[t][i] *= forwardBackward.model.param.observation
									.get(k).get(words[t][k], i);
						}
					}
				}
			}
		}
		return observationCache[position][state];

	}

	/*
	 * returns the original word at the position
	 */
	public String getWord(int position) {
		return c.corpusVocab.get(0).indexToWord.get(words[position][0]);
	}

	public void populateWordArray(String line) {
		String allTimeSteps[] = line.split(c.delimiter);
		T = allTimeSteps.length;
		words = new int[T][c.oneTimeStepObsSize];
		for (int i = 0; i < T; i++) {
			String oneTimeStep = allTimeSteps[i];
			String[] obsElements = oneTimeStep.split(c.obsDelimiter);
			// original word
			String word = obsElements[0];
			if (c.corpusVocab.get(0).lower) {
				word = word.toLowerCase();
			}
			if (c.corpusVocab.get(0).smooth) {
				word = SmoothWord.smooth(word);
			}
			int wordId = c.corpusVocab.get(0).getIndex(word);
			words[i][0] = wordId;
			// for hmm states as observations
			for (int j = 1; j < obsElements.length; j++) {
				String obsElement = obsElements[j];
				int obsElementId = c.corpusVocab.get(j).getIndex(obsElement);
				words[i][j] = obsElementId;
			}
		}
	}
}
