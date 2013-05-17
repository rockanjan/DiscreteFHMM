package model.train;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

public class OptimizerExample implements Optimizable.ByGradientValue {

    // Optimizables encapsulate all state variables, 
    //  so a single Optimizer object can be used to optimize 
    //  several functions.

    double[] parameters;

    public OptimizerExample(double[] init) {
        parameters = new double[2];
        parameters[0] = init[0];
        parameters[1] = init[1];
    }

    public double getValue() {

        double x = parameters[0];
        double y = parameters[1];

        return -3*x*x - 4*y*y + 2*x - 4*y + 18;

    }

    public void getValueGradient(double[] gradient) {

        gradient[0] = -6 * parameters[0] + 2;
        gradient[1] = -8 * parameters[1] - 4;

    }

    // The following get/set methods satisfy the Optimizable interface

    public int getNumParameters() { return 2; }
    public double getParameter(int i) { return parameters[i]; }
    public void getParameters(double[] buffer) {
        buffer[0] = parameters[0];
        buffer[1] = parameters[1];
    }

    public void setParameter(int i, double r) {
        parameters[i] = r;
    }
    public void setParameters(double[] newParameters) {
        parameters[0] = newParameters[0];
        parameters[1] = newParameters[1];
    }
    
    public static void main(String[] args) {
    	//double[] initParams = {10000, 100000}; //messes up
    	double[] initParams = {10, 100};
    	OptimizerExample optimizable = new OptimizerExample(initParams);
        Optimizer optimizer = new LimitedMemoryBFGS(optimizable);

        boolean converged = false;

        try {
            converged = optimizer.optimize();
        } catch (IllegalArgumentException e) {
            // This exception may be thrown if L-BFGS
            //  cannot step in the current direction.
            // This condition does not necessarily mean that
            //  the optimizer has failed, but it doesn't want
            //  to claim to have succeeded...        
        }

        System.out.println(optimizable.getParameter(0) + ", " +
                           optimizable.getParameter(1));
    }
}