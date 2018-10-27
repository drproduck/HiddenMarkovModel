package em.map;

import java.util.*;

/**
 * bigram em.HMM
 * em.EM algorithm. Emission has a sparse-inducing prior
 */
public class MapHMM {

    public double[][][] digamma;
    public double[][] A, B, ap, bt, gamma;
    public double[] c, pi;
    public int[] seq;
    public MDir dirPrior;
    double logProb;
    double oldLogProb;
    boolean firstRestart;

    public int len, nt, nw;

    public MapHMM(int n_states, int n_obs, int[] observations, MDir dirPrior) {
        this(n_states, n_obs, observations.length, dirPrior);
        seq = observations;
    }

    public MapHMM(int n_states, int n_obs, int time, MDir dirPrior) {
        len = time;
        this.nt = n_states;
        this.nw = n_obs;
        A = new double[nt][nt];
        B = new double[nt][nw];
        pi = new double[nt];
        ap = new double[len][nt];
        bt = new double[len][nt];
        c = new double[len];
        gamma = new double[len][nt];
        digamma = new double[len][nt][nt];
        this.dirPrior = dirPrior;
        firstRestart = true;
    }

    public void alpha() {
        c[0] = 0;

        for (int i = 0; i < nt; i++) {
            ap[0][i] = pi[i] * B[i][seq[0]];
            c[0] += ap[0][i];
        }

        c[0] = 1/ c[0];
        for (int i = 0; i < nt; i++) {
            ap[0][i] = c[0] * ap[0][i];
        }

        for (int t = 1; t < len; t++) {
            c[t] = 0;
            for (int i = 0; i < nt; i++) {
                ap[t][i] = 0;
                for (int j = 0; j < nt; j++) {
                    ap[t][i] += ap[t - 1][j] * A[j][i];
                }
                ap[t][i] = ap[t][i] * B[i][seq[t]];
                c[t] += ap[t][i];
            }
            //scale
            c[t] = 1/c[t];
            for (int i = 0; i < nt; i++) {
                ap[t][i] *= c[t];
            }
        }
    }

    //space optimized, self-contained.
    public int[] viterbi() {
        double c = 0;

        int[][] dp = new int[len][nt];
        double[] head = new double[nt];
        double[] temp = new double[nt];

        for (int i = 0; i < nt; i++) {
            head[i] = pi[i] * Math.pow(B[i][seq[0]], 3);
            c += ap[0][i];
        }

        c = 1 / c;
        for (int i = 0; i < nt; i++) {
            head[i] *= c;
        }

        for (int t = 1; t < len; t++) {
            c = 0;
            for (int i = 0; i < nt; i++) {
                double x = 0;
                double max = Double.NEGATIVE_INFINITY;
                int lst = -1;
                for (int j = 0; j < nt; j++) {
                    x = head[j] * A[j][i];
                    if (x > max) {
                        max = x;
                        lst = j;
                    }
                }
                dp[t][i] = lst;

                temp[i] = max * Math.pow(B[i][seq[t]],3);
                c += temp[i];
            }
            //scale
            c = 1/c;
            for (int i = 0; i < nt; i++) {
                temp[i] *= c;
            }

            //copy temp back to head
            for (int i = 0; i < nt; i++) {
                head[i] = temp[i];
            }
        }

        //get the last state of the best DP sequence
        int bs = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < nt; i++) {
            if (head[i] > max) {
                max = head[i];
                bs = i;
            }
        }
        //trace back
        int[] res = new int[len];
        res[len - 1] = bs;
        for (int t = len -1; t > 0; t--) {
            bs = dp[t][bs];
            res[t-1] = bs;
        }

        return res;
    }

    // best sequence in em.HMM sense
    public int[] besthmm() {
        alpha(); beta(); gamma();
        int[] res = new int[len];
        for (int t = 0; t < len; t++) {
            double max = Double.NEGATIVE_INFINITY;
            int p = -1;
            for (int i = 0; i < nt; i++) {
                if (gamma[t][i] > max) {
                    max = gamma[t][i];
                    p = i;
                }
            }
            res[t] = p;
        }
        return res;
    }

    public void beta() {
        // Let βT −1(i) = 1, scaled by cT −1
        for (int i = 0; i < nt; i++) {
            bt[len - 1][i] = c[len - 1];
        }
        for (int t = len - 2; t >= 0; t--) {
            for (int i = 0; i < nt; i++) {
                bt[t][i] = 0;
                for (int j = 0; j < nt; j++) {
                    bt[t][i] = bt[t][i] + A[i][j] * B[j][seq[t + 1]] * bt[t + 1][j];
                }
                // scale βt(i) with same scale factor as αt(i)
                bt[t][i] = c[t] * bt[t][i];
            }
        }
    }

    public void gamma() {
        for (int t = 0; t < len - 1; t++) {
            double denom = 0;
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    denom = denom + ap[t][i] * A[i][j] * B[j][seq[t + 1]] * bt[t + 1][j];
                }
            }
            for (int i = 0; i < nt; i++) {
                gamma[t][i] = 0;
                for (int j = 0; j < nt; j++) {
                    digamma[t][i][j] = (ap[t][i] * A[i][j] * B[j][seq[t+1]] * bt[t+1][j]) / denom;
                    gamma[t][i] = gamma[t][i] + digamma[t][i][j];
                }
            }
        }

        // special case
        double denom = 0;
        for (int i = 0; i < nt; i++) {
            denom = denom + ap[len - 1][i];
        }
        for (int i = 0; i < nt; i++) {
            gamma[len - 1][i] = ap[len - 1][i] / denom;
        }
    }


    //reEstimate the model only after c, beta and gamma passes
    public void reEstimate(boolean trainA) {
        if (firstRestart) {
            alpha();
            logProbFromAlpha();
            oldLogProb = logProb;
            firstRestart = false;
        }
        beta();
        gamma();
        for (int i = 0; i < nt; i++) {
            pi[i] = gamma[0][i];
        }

        if (trainA) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    double numer = 0;
                    double denom = 0;
                    for (int t = 0; t < len - 1; t++) {
                        numer = numer + digamma[t][i][j];
                        denom = denom + gamma[t][i];
                    }
                    A[i][j] = numer / denom;
                }
            }
        }

        for (int i = 0; i < nt; i++) {
            double[] expectedCount = new double[nw];
            for (int j = 0; j < nw; j++) {
                for (int t = 0; t < len; t++) {
                    if (seq[t] == j) {
                        expectedCount[j] += gamma[t][i];
                    }
                }
            }
            B[i] = dirPrior.getMode(expectedCount);
        }

        alpha();
        logProbFromAlpha();
    }

    // compute marginalized log probability P(x | lambda)
    public double logProbFromAlpha() {
        logProb = 0;
        for (int i = 0; i < len; i++) {
            logProb = logProb + Math.log(c[i]);
        }
        logProb = -logProb;
        return logProb;
    }

    // marginalized log probability given an observation sequence
    public double logProbFromAlpha(int[] obs) {
        seq = obs;
        alpha();
        double p = 1;
        for (int i = 0; i < c.length; i++) {
            p *= c[i];
        }
        return 1.0/p;
    }

    public void train(int iter, long seed, boolean trainA, boolean verbose) {
        init(seed, trainA);
        long start;
        long stop;
        for (int it = 0; it < iter; it++) {
            if (verbose) System.out.printf("iteration: %d\n", it);
            start = System.nanoTime();
            reEstimate(trainA);
            stop = System.nanoTime();
            if (verbose) {
                System.out.println(logProb);
                System.out.printf("time: %f\n", (stop - start) / 1e9);
            }
            if (it != 0 && logProb <= oldLogProb) {
                System.out.printf("something may be wrong: logprob = %f, oldlogprob = %f\n", logProb, oldLogProb);
                break;
            }
        }
    }

    // compute model joint probability P(x, z | lambda)
    public double getJointProb(int[] states) throws Exception {
        if (states.length != len) throw new Exception("number of states must correspond to number of observations");
        double p = 1;
        p *= pi[states[0]] * B[states[0]][seq[0]];
        for (int t = 1; t < len; t++) {
            p *= A[states[t - 1]][states[t]] * B[states[t]][seq[t]];
        }
        return p;
    }

    // randomly initialize pi, A and B so that the rows sum to 1

    public void init(long seed, boolean initA) {
        Random rng = new Random(seed);
        double rem = 1.0;
        for (int i = 0; i < nt -1; i++) {
            pi[i] = rng.nextDouble() * rem;
            rem = rem - pi[i];
        }
        pi[nt -1] = rem;

        if (initA) {
            for (int i = 0; i < nt; i++) {
                rem = 1.0;
                for (int j = 0; j < nt - 1; j++) {
                    A[i][j] = rng.nextDouble() * rem;
                    rem = rem - A[i][j];
                }
                A[i][nt - 1] = rem;
            }
        }

        for (int i = 0; i < nt; i++) {
            rem = 1.0;
            for (int j = 0; j < nw -1; j++) {
                B[i][j] = rng.nextDouble() * rem;
                rem = rem - B[i][j];
            }
            B[i][nw -1] = rem;
        }
    }

    double sc = 0;

    // code to backtrack in problem 3
    public void prob3(int n, int l) {
        sc = 0;
        int po = 0;
        int[] ar = new int[l];
        for (int i = 0; i < n; i++) {
            ar[po] = i;
            bt(n, l, po+1, ar);
            ar[po] = 0;
        }
    }

    private void bt(int n, int l, int po, int[] ar) {
        if (po == l){
            System.out.print(Arrays.toString(ar)+" ");
            double cursc = logProbFromAlpha(ar);
            System.out.println(cursc);
            sc += cursc;
            return;
        }
        for (int i = 0; i < n; i++) {
            ar[po] = i;
            bt(n, l, po+1, ar);
            ar[po] = 0;
        }
    }
}


