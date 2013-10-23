package model;

import java.util.HashMap;
import java.util.Map;

import config.Config;

public class StateCombination {
	public int[] states;
	public StateCombination(int[] states) {
		this.states = states;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for(int i=0; i<states.length; i++) {
			sb.append(states[i] + " ");
		}
		return sb.toString();
	}
}
