package em;

import em.ExpectationEstimator;
import em.Settings;
import org.apache.commons.math3.random.RandomGenerator;

public class EM {

    public double[][] B;
    int len;
    int nt;
    int nw;
    double[][] expectation;
    int[] seq;
    double logProb;
    double oldLogProb;
    boolean firstRestart;
    static double eps = 0.1; // smoothing constant
    Settings settings;
    RandomGenerator rng;
    ExpectationEstimator estimator;

    public EM(int nTag, int nWord, int[] obs, ExpectationEstimator estimator, RandomGenerator rng, Settings settings) {
        len = obs.length;
        nt = nTag;
        nw = nWord;
        B = new double[nt][nw];
        expectation = new double[len][nt];
        seq = obs;
        this.settings = settings;
        this.rng = rng;
        this.estimator = estimator;
        firstRestart = true;
    }

    public void maximization(double[][] exp) {

        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nw; j++) {
                double denom = eps * nw;
                double numer = 0;
                for (int t = 0; t < len; t++) {
                    if (seq[t] == j) {
                        numer += exp[t][i];
                    }
                    denom += exp[t][i];
                }
                numer += eps;
                B[i][j] = numer / denom;
            }
        }
    }

    public void train(int iter) {
        long start;
        long stop;
        initB();
        for (int it = 0; it < iter; it++) {
            if (settings.verbose) System.out.printf("iteration: %d\n", it);

            start = System.nanoTime();
            estimator.setB(B);
            double[][] exp = estimator.estimate();
            maximization(exp);
            stop = System.nanoTime();
            if (settings.verbose) {
                System.out.printf("time: %f\n", (stop - start) / 1e9);
            }
//            if (it != 0 && logProb <= oldLogProb) {
//                System.out.printf("something may be wrong: logprob = %f, oldlogprob = %f\n", logProb, oldLogProb);
//                break;
//            }
        }
    }

    public void initB() {
        double s;
        for (int i = 0; i < nt; i++) {
            s = 0;
            for (int j = 0; j < nw; j++) {
                B[i][j] = rng.nextDouble();
                s += B[i][j];
            }
            for (int j = 0; j < nw; j++) {
                B[i][j] /= s;
            }
        }
    }

    public double[][] getB(){
        return B;
    }
}
