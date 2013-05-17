package model.train;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

public class ExampleOptimizable implements Optimizable.ByGradientValue{
	
	double[] parameters;
	
	boolean paramChanged = true;
	double latestValue = 0.0;
	double[] latestGradient;
	
	public ExampleOptimizable(double[] initParams) {
		parameters = new double[initParams.length];
		for(int i=0; i<initParams.length; i++) {
			parameters[i] = initParams[i];
		}
		latestGradient = new double[parameters.length];
	}
	
	@Override
	public double getValue() {
		//System.out.println("get value called");
		if(paramChanged) {
			double x = parameters[0];
	        double y = parameters[1];
	        latestValue = -3*x*x - 4*y*y + 2*x - 4*y + 18;
		}
        return latestValue;
	}

	@Override
	public void getValueGradient(double[] gradient) {
		if(paramChanged) {
			gradient[0] = -6 * parameters[0] + 2;
	        gradient[1] = -8 * parameters[1] - 4;
	        for(int i=0; i<parameters.length; i++) {
				latestGradient[i] = gradient[i];
			}
		} else {
			for(int i=0; i<parameters.length; i++) {
				gradient[i] = latestGradient[i];
			}
		}
	}

	@Override
	public int getNumParameters() {
		return parameters.length;
	}

	@Override
	public double getParameter(int i) {
		return parameters[i];
	}

	@Override
	public void getParameters(double[] buffer) {
		for(int i=0; i<parameters.length; i++) {
			buffer[i] = parameters[i];
		}
		
	}

	@Override
	public void setParameter(int i, double value) {
		System.out.println("set parameter called");
		parameters[i] = value;
		paramChanged = true;
	}

	@Override
	public void setParameters(double[] newParam) {
		System.out.println("set parameters called");
		for(int i=0; i<parameters.length; i++) {
			parameters[i] = newParam[i];
		}
		paramChanged = true;
	}
	
	public double myGetValue(double x, double y) {
		return -3*x*x - 4*y*y + 2*x - 4*y + 18;
	}
	
	
	public static void main(String[] args) {
		//double[] initParams = {0.33, -0.5}; //the solution
		//double[] initParams = {33e100, 5e-100}; //messes up
		//double[] initParams = {100, 1000000}; //messes up
		double[] initParams = {10000, 100000}; //messes up

		//double[] initParams = {1,1};
		ExampleOptimizable optimizable = new ExampleOptimizable(initParams);
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		//System.out.println(optimizable.myGetValue(90.024, 899899.00));
		//System.out.println(optimizable.myGetValue(.33, -.5));
		//System.exit(-1);
		boolean converged = false;
		 try {
	            converged = optimizer.optimize();
	        } catch (IllegalArgumentException e) {
	            System.out.println("optimization throw exception");        
	        }

	        System.out.println(optimizable.getParameter(0) + ", " +
	                           optimizable.getParameter(1));
	}
}
