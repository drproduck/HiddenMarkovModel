import java.util.Random;

/**
 * Created by drproduck on 11/19/17.
 */
public class BayesianTriHMM {
    public double[][][] digamma;
    public double[][] A, B, alp, bet, gamma;
    public double[] c, pi;
    public int[] O;
    public MDir dirPrior;

    public int T, N, M;

    public BayesianTriHMM(int n_states, int n_obs, int[] observations, MDir dirPrior) {
        this(n_states, n_obs, observations.length, dirPrior);
        O = observations;
    }

    public BayesianTriHMM(int n_states, int n_obs, int time, MDir dirPrior) {
        T = time;
        this.N = n_states;
        this.M = n_obs;
        A = new double[N][N];
        B = new double[N][M];
        pi = new double[N];
        alp = new double[T][N];
        bet = new double[T][N];
        c = new double[T];
        gamma = new double[T][N];
        digamma = new double[T][N][N];
        this.dirPrior = dirPrior;
    }

    public void alpha() {
        c[0] = 0;

        for (int i = 0; i < N; i++) {
            alp[0][i] = pi[i] * B[i][O[0]];
            c[0] += alp[0][i];
        }

        c[0] = 1 / c[0];
        for (int i = 0; i < N; i++) {
            alp[0][i] = c[0] * alp[0][i];
        }

        for (int t = 1; t < T; t++) {
            c[t] = 0;
            for (int i = 0; i < N; i++) {
                alp[t][i] = 0;
                for (int j = 0; j < N; j++) {
                    alp[t][i] += alp[t - 1][j] * A[j][i];
                }
                alp[t][i] = alp[t][i] * B[i][O[t]];
                c[t] += alp[t][i];
            }
            //scale
            c[t] = 1 / c[t];
            for (int i = 0; i < N; i++) {
                alp[t][i] *= c[t];
            }
        }
    }

    //space optimized, self-contained.
    public int[] bestDP() {
        double c = 0;

        int[][] dp = new int[T][N];
        double[] head = new double[N];
        double[] temp = new double[N];

        for (int i = 0; i < N; i++) {
            head[i] = pi[i] * B[i][O[0]];
            c += alp[0][i];
        }

        c = 1 / c;
        for (int i = 0; i < N; i++) {
            head[i] *= c;
        }

        for (int t = 1; t < T; t++) {
            c = 0;
            for (int i = 0; i < N; i++) {
                double x = 0;
                double max = Double.NEGATIVE_INFINITY;
                int lst = -1;
                for (int j = 0; j < N; j++) {
                    x = head[j] * A[j][i];
                    if (x > max) {
                        max = x;
                        lst = j;
                    }
                }
                dp[t][i] = lst;

                temp[i] = max * B[i][O[t]];
                c += temp[i];
            }
            //scale
            c = 1 / c;
            for (int i = 0; i < N; i++) {
                temp[i] *= c;
            }

            //copy temp back to head
            for (int i = 0; i < N; i++) {
                head[i] = temp[i];
            }
        }

        //get the last state of the best DP sequence
        int bs = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            if (head[i] > max) {
                max = head[i];
                bs = i;
            }
        }
        //trace back
        int[] res = new int[T];
        res[T - 1] = bs;
        for (int t = T - 1; t > 0; t--) {
            bs = dp[t][bs];
            res[t - 1] = bs;
        }

        return res;
    }

    // best sequence in HMM sense
    public int[] besthmm() {
        alpha();
        beta();
        gamma();
        int[] res = new int[T];
        for (int t = 0; t < T; t++) {
            double max = Double.NEGATIVE_INFINITY;
            int p = -1;
            for (int i = 0; i < N; i++) {
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
        for (int i = 0; i < N; i++) {
            bet[T - 1][i] = c[T - 1];
        }
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                bet[t][i] = 0;
                for (int j = 0; j < N; j++) {
                    bet[t][i] = bet[t][i] + A[i][j] * B[j][O[t + 1]] * bet[t + 1][j];
                }
                // scale βt(i) with same scale factor as αt(i)
                bet[t][i] = c[t] * bet[t][i];
            }
        }
    }

    public void gamma() {
        for (int t = 0; t < T - 1; t++) {
            double denom = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    denom = denom + alp[t][i] * A[i][j] * B[j][O[t + 1]] * bet[t + 1][j];
                }
            }
            for (int i = 0; i < N; i++) {
                gamma[t][i] = 0;
                for (int j = 0; j < N; j++) {
                    digamma[t][i][j] = (alp[t][i] * A[i][j] * B[j][O[t + 1]] * bet[t + 1][j]) / denom;
                    gamma[t][i] = gamma[t][i] + digamma[t][i][j];
                }
            }
        }

        double denom = 0;
        for (int i = 0; i < N; i++) {
            denom = denom + alp[T - 1][i];
        }
        for (int i = 0; i < N; i++) {
            gamma[T - 1][i] = alp[T - 1][i] / denom;
        }
    }

    //reEstimate the model only after c, beta and gamma passes
    public void train(boolean trainA) {
        for (int i = 0; i < N; i++) {
            pi[i] = gamma[0][i];
        }

        if (trainA) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double numer = 0;
                    double denom = 0;
                    for (int t = 0; t < T - 1; t++) {
                        numer = numer + digamma[t][i][j];
                        denom = denom + gamma[t][i];
                    }
                    A[i][j] = numer / denom;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            double[] expectedCount = new double[M];
            for (int j = 0; j < M; j++) {
                for (int t = 0; t < T; t++) {
                    if (O[t] == j) {
                        expectedCount[j] += gamma[t][i];
                    }
                }
            }
            B[i] = dirPrior.getMode(expectedCount);
        }
    }

    // compute marginalized log probability P(x | lambda)
    public double getLogProb() {
        double logProb = 0;
        for (int i = 0; i < T; i++) {
            logProb = logProb + Math.log(c[i]);
        }
        return -logProb;
    }

    // marginalized log probability given an observation sequence
    public double getLogProb(int[] obs) {
        O = obs;
        alpha();
        double p = 1;
        for (int i = 0; i < c.length; i++) {
            p *= c[i];
        }
        return 1.0 / p;
    }

    // actual reEstimate algorithm
    public double iterate(int maxIters, long seed, boolean trainA, boolean initA, boolean verbose) {
        double logProb = Double.NEGATIVE_INFINITY;
        double oldLogProb = Double.NEGATIVE_INFINITY;
        init(seed, initA);
        int iters = 0;
        alpha();
        beta();
        gamma();
        train(trainA);
        logProb = getLogProb();
        if (verbose) System.out.println(logProb);
        iters++;
        while (iters < maxIters && logProb > oldLogProb) {
            oldLogProb = logProb;
            alpha();
            beta();
            gamma();
            train(trainA);
            logProb = getLogProb();
            if (verbose) System.out.println(logProb);
            iters++;
        }
        return oldLogProb;
    }

    // compute model joint probability P(x, z | lambda)
    public double getJointProb(int[] states) throws Exception {
        if (states.length != T) throw new Exception("number of states must correspond to number of observations");
        double p = 1;
        p *= pi[states[0]] * B[states[0]][O[0]];
        for (int t = 1; t < T; t++) {
            p *= A[states[t - 1]][states[t]] * B[states[t]][O[t]];
        }
        return p;
    }

    // randomly initialize pi, A and B so that the rows sum to 1

    public void init(long seed, boolean initA) {
        Random rng = new Random(seed);
        double rem = 1.0;
        for (int i = 0; i < N - 1; i++) {
            pi[i] = rng.nextDouble() * rem;
            rem = rem - pi[i];
        }
        pi[N - 1] = rem;

        if (initA) {
            for (int i = 0; i < N; i++) {
                rem = 1.0;
                for (int j = 0; j < N - 1; j++) {
                    A[i][j] = rng.nextDouble() * rem;
                    rem = rem - A[i][j];
                }
                A[i][N - 1] = rem;
            }
        }

        for (int i = 0; i < N; i++) {
            rem = 1.0;
            for (int j = 0; j < M - 1; j++) {
                B[i][j] = rng.nextDouble() * rem;
                rem = rem - B[i][j];
            }
            B[i][M - 1] = rem;
        }
    }

}
