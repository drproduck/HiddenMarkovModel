package em;

import org.apache.commons.math3.random.RandomGenerator;

public class ForwardBackwardEstimator extends ExpectationEstimator {

    double[][][] A;
    double[][][] ap, bt;
    double[] c;
    double[][][] digamma;
    double[][] gamma;

    public ForwardBackwardEstimator(double[][][] transition, int nTag, int nWord, int[] obs, RandomGenerator rng, Settings settings) {
        super(nTag, nWord, obs, rng, settings);
        A = transition;
        ap = new double[len - 1][nt][nt];
        bt = new double[len - 1][nt][nt];
        c = new double[len - 1];
        digamma = new double[len][nt][nt];
        gamma = new double[len][nt];
    }

    public void alpha() {
        c[0] = 0;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                ap[0][i][j] = B[i][obs[0]] * B[j][obs[1]];
                c[0] += ap[0][i][j];
            }
        }
        c[0] = 1 / c[0];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                ap[0][i][j] *= c[0];
            }
        }
        for (int t = 1; t < len - 1; t++) {
            c[t] = 0;
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    ap[t][i][j] = 0;
                    for (int k = 0; k < nt; k++) {
                        ap[t][i][j] += ap[t - 1][k][i] * A[k][i][j] * B[j][obs[t + 1]];
                    }
                    c[t] += ap[t][i][j];
                }
            }
            c[t] = 1 / c[t];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    ap[t][i][j] *= c[t]; // p(O0,O1,x0,x1) / p(O0,O1) = p(x0,x1|O0,O1)
                }
            }
        }
    }

    public void beta() {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                bt[len - 2][i][j] = 1;
            }
        }
        for (int t = len - 3; t >= 0; t--) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    bt[t][i][j] = 0;
                    for (int k = 0; k < nt; k++) {
                        bt[t][i][j] += bt[t + 1][j][k] * A[i][j][k] * B[k][obs[t + 2]];
                    }
                    bt[t][i][j] *= c[t + 1]; // p(O2|x0,x1) / p(O2|O0,O1)
                }
            }
        }
    }

    public double[][] estimate() {
        alpha();
        beta();
        for (int t = 0; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                gamma[t][i] = 0;
                for (int j = 0; j < nt; j++) {
                    digamma[t][i][j] = ap[t][i][j] * bt[t][i][j]; // p(O0,01,O2,x0,x1) / p(O0,O1,O2) = p(x0,x1|O)
                    gamma[t][i] += digamma[t][i][j]; // p(x0|O) = \sum_{x1} p(x0,x1|O)
                }
            }
        }
        for (int j = 0; j < nt; j++) {
            gamma[len - 1][j] = 0;
            for (int i = 0; i < nt; i++) {
                gamma[len - 1][j] += digamma[len - 2][i][j]; // p(x2|O) = \sum_{x1} p(x1,x2|O)
            }
        }
        return gamma;
    }
}
