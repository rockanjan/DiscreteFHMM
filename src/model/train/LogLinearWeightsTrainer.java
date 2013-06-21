package model.train;

import corpus.Corpus;

public abstract class LogLinearWeightsTrainer {
	Corpus corpus;
	
	public LogLinearWeightsTrainer(Corpus c) {
		this.corpus = c;
	}
	
	public abstract void train();
}
