package util;

import java.text.DecimalFormat;
import java.util.Date;

public class Timing {
	private long startT;
	private long stopT;
	private DecimalFormat df = new DecimalFormat("##.####");
	public Timing() {
		startT = new Date().getTime();
	}
	public void start() {
		//startT = System.currentTimeMillis();
		startT = new Date().getTime();
	}
	
	/*
	 * return time elapsed since start() called
	 */
	public String stop() {
		//stopT = System.currentTimeMillis();
		stopT = new Date().getTime();
		String elapsed = df.format((1.0 * (stopT - startT) / 1000 / 60))  + " minutes";
		stopT = 0; startT = 0;
		return elapsed;
		
	}
	
	public long stopGetLong() {
		//stopT = System.currentTimeMillis();
		stopT = new Date().getTime();
		long elapsed = (stopT - startT);
		stopT = 0; startT = 0;
		return elapsed;
		
	}
	
	public String stopMilliseconds() {
		//stopT = System.currentTimeMillis();
		stopT = new Date().getTime();
		String elapsed = (stopT - startT)  + " milliseconds";
		stopT = 0; startT = 0;
		return elapsed;
		
	}
	
}
