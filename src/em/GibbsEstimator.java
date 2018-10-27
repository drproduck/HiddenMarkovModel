package em;

import math.CategoricalDistribution;
import math.Counters;
import math.DirichletDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import util.Helper;

import java.util.Arrays;

public class GibbsEstimator extends ExpectationEstimator {

    double[][][] A;

    public GibbsEstimator(double[][][] transition, int nTag, int nWord, int[] obs, RandomGenerator rng, Settings settings) {
        super(nTag, nWord, obs, rng, settings);
        A = transition;
    }

    public void sample(int[] hidden){
        for (int t = 0; t < len; t++) {
            double[] P = new double[nt];
            double s = 0;

            if (t == 0) {
                for (int i = 0; i < nt; i++) {
                    P[i] = A[i][hidden[1]][hidden[2]] * B[i][obs[t]];
                    s += P[i];
                }
            } else if (t == 1) {
                for (int i = 0; i < nt; i++) {
                    P[i] = A[hidden[0]][i][hidden[2]] * A[i][hidden[2]][hidden[3]] * B[i][obs[t]];
                    s += P[i];
                }
            } else if (t == len - 1) {
                for (int i = 0; i < nt; i++) {
                    P[i] = A[hidden[len - 3]][hidden[len - 2]][i] * B[i][obs[t]];
                    s += P[i];
                }
            } else if (t == len - 2) {
                for (int i = 0; i < nt; i++) {
                    P[i] = A[hidden[len - 4]][hidden[len - 3]][i] * A[hidden[len - 3]][i][hidden[len - 1]] * B[i][obs[t]];
                    s += P[i];
                }
            } else {
                for (int i = 0; i < nt; i++) {
                    P[i] = A[hidden[t - 2]][hidden[t - 1]][i] * A[hidden[t - 1]][i][hidden[t + 1]] * A[i][hidden[t + 1]][hidden[t + 2]] * B[i][obs[t]];
                    s += P[i];
                }
            }
            for (int i = 0; i < nt; i++) {
                P[i] /= s;
            }
            hidden[t] = cd.sample(P);
        }
    }

    public double[][] estimate(){

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

//            logProb[it] = logProb(hidden);
            stop = System.nanoTime();
            if (settings.verbose) {
                System.out.println("iteration "+it);
//                System.out.println(logProb[it]);
//                System.out.println(Arrays.toString(Helper.toCharacterSequence(hidden)));
//                System.out.printf("time: %f\n", (stop - start) / 1e9);
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
