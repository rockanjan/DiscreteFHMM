package model.inference;

import java.util.Random;

import program.Main;

public class VariationalParamAlpha {
	double alpha;
	
	public void initializeRandom() {
		Random r = new Random(Main.seed);
		alpha = r.nextDouble();
	}
}
