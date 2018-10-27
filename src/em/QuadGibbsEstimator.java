package em;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import math.Math;
import util.Helper;

import java.util.Arrays;
import java.util.stream.DoubleStream;


public class QuadGibbsEstimator extends ExpectationEstimator {

    double[][][][] A;

    public QuadGibbsEstimator(double[][][][] transition, int nTag, int nWord, int[] obs, RandomGenerator rng, Settings settings) {
        super(nTag, nWord, obs, rng, settings);
        A = transition;
    }

    public void sample(int[] hidden) {
        for (int t = 0; t < len; t++) {
            double[] P = new double[nt];

            if (t==0) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[i][hidden[1]][hidden[2]][hidden[3]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else if (t==1) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[0]][i][hidden[2]][hidden[3]]) +
                            FastMath.log(A[i][hidden[2]][hidden[3]][hidden[4]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else if (t==2) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[0]][hidden[1]][i][hidden[3]]) +
                            FastMath.log(A[hidden[1]][i][hidden[3]][hidden[4]]) +
                            FastMath.log(A[i][hidden[3]][hidden[4]][hidden[5]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else if (t==len-1) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[len-4]][hidden[len-3]][hidden[len-2]][i]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else if (t==len-2) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[len - 5]][hidden[len - 4]][hidden[len - 3]][i]) +
                            FastMath.log(A[hidden[len - 4]][hidden[len - 3]][i][hidden[len - 1]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else if (t==len-3) {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[len-6]][hidden[len-5]][hidden[len-4]][i]) +
                            FastMath.log(A[hidden[len-5]][hidden[len-4]][i][hidden[len-2]]) +
                            FastMath.log(A[hidden[len-4]][i][hidden[len-2]][hidden[len-1]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            else {
                for (int i = 0; i < nt; i++) {
                    P[i] = FastMath.log(A[hidden[t-3]][hidden[t-2]][hidden[t-1]][i]) +
                            FastMath.log(A[hidden[t-2]][hidden[t-1]][i][hidden[t+1]]) +
                            FastMath.log(A[hidden[t-1]][i][hidden[t+1]][hidden[t+2]]) +
                            FastMath.log(A[i][hidden[t+1]][hidden[t+2]][hidden[t+3]]) +
                            FastMath.log(B[i][obs[t]]);
                }
            }
            hidden[t] = choice(P);
        }
    }

    public int choice(double[] logWeights){
        double logSum = Math.logSumExp(logWeights);
        double[] cumweights = new double[logWeights.length];
        cumweights[0] = FastMath.exp(logWeights[0] - logSum);
        for (int i = 1; i < logWeights.length; i++) {
            cumweights[i] = FastMath.exp(logWeights[i] - logSum) + cumweights[i-1];
        }
        double val = rng.nextDouble() * cumweights[cumweights.length-1];
        return java.lang.Math.abs(Arrays.binarySearch(cumweights, val)) - 1;
    }

    public double[][] estimate() {

        logProb = new double[settings.nIter];
        long start;
        long stop;

        double[][] expectation = new double[len][nt];

        int[] hidden = firstSample();
        for (int it = 0; it < settings.nIter; it++) {
            if (settings.verbose) System.out.printf("iteration: %d\n", it);
            start = System.nanoTime();

            sample(hidden);

            if (it >= settings.burnin) {
                for (int t = 0; t < len; t++) {
                    expectation[t][hidden[t]] += 1;
                }
            }

//            logProb[it] = logProb(hidden, A);
            stop = System.nanoTime();
            if (settings.verbose) {
                System.out.println("iteration "+it);
//                System.out.println(logProb[it]);
                System.out.println(Arrays.toString(Helper.toCharacterSequence(hidden)));
                System.out.printf("time: %f\n", (stop - start) / 1e9);
            }
        }

        for (int t = 0; t < len; t++) {
            for (int i = 0; i < nt; i++) {
                expectation[t][i] /= (settings.nIter - settings.burnin);
            }
        }
        return expectation;
    }
}
