import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import java.util.*;

/**
 * Trigram HMM. a similar normalization is used instead of log-scale and the ensuing logSumExp.
 */

public class HMMBlockSampler {

    double[][][] A;
    double[][] B;
    int len;
    int nt;
    int nw;
    double[][][] alpha;
    double[] c;
    double[][][][] trigamma;
    int[] obs;
    double[] logProb;
    int[] hiddenSample;

    CategoricalDistribution cd;
    DirichletDistribution dirichet;
    RandomGenerator rng = new MersenneTwister();
    static double eps = 0.1; // smoothing constant

    Counters<Integer>[] empiricalCounts;

    public HMMBlockSampler(double[][][] transition, int nTag, int nWord, int[] observations) {
        len = observations.length;
        nt = nTag;
        nw = nWord;
        A = transition;
        B = new double[nt][nw];
        alpha = new double[len - 1][nt][nt];
        c = new double[len - 1];
        trigamma = new double[len][nt][nt][nt];

        obs = observations;

        cd = new CategoricalDistribution(rng);
        double[] prior = new double[nw];
        Arrays.fill(prior, 1.0 / nw);
        assert(prior[0] == 1.0 / nw);
        dirichet = new DirichletDistribution(rng, prior);

       empiricalCounts = new Counters[nt];
       Arrays.fill(empiricalCounts, new Counters<Integer>());
    }

    /*
    Step 1:
    Compute p(x_t = q, x_{t+1} = r, x_{x+2} = s | O_0^{t+2})
    = p(x_t = q, x_{t+1} = r, x_{x+2} = s, O_{t+2} | O_0^{t+1}) / p(O_{t+2} | O_0^{t+1})
    = p(x_{t}, x_{x+1} | O_0^{t+1}) * p(x_{t+2} | x_{t}, x_{t+1}) * p(O_{t+2} | x_{t+2})

    Step 2:
    compute p(x_{t+1} = r, x_{t+2} = s | O_0^{t+2}) for recursion (prior)

    Step 3:
    Sample p(\bold{x}|O) = p(x_{n}, x_{n-1} | O_0^n) \prod p(x_{t} | x_{t+1}^n, O_0^n)
                         =                            prod p(x_{t} | x_{t+1}, x_{t+2}, O_0^{t+2})
    where p(x_{t} | x{t+1}, x_{t+2}, O_0^{t+2}) ~ p(x_{t}, x{t+1}, x{t+2} | O_0^{t+2}) (beta)
     */

    public void forward() {

        c[0] = 0;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                alpha[0][i][j] = B[i][obs[0]] * B[j][obs[1]];
                c[0] += alpha[0][i][j];
            }
        }
        c[0] = 1 / c[0];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                alpha[0][i][j] *= c[0];
            }
        }

        for (int t = 1; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        trigamma[t][i][j][k] = alpha[t-1][i][j] * A[i][j][k] * B[k][obs[t+1]];
                        alpha[t][j][k] += trigamma[t][i][j][k];
                        c[t] += alpha[t][j][k];
                    }
                }
            }
            // normalize
            c[t] = 1 / c[t];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    alpha[t][i][j] *= c[t];
                }
            }
        }
    }

    public int[] backwardSampling() {
        int[] sample = new int[len];
        Pair<Integer, Integer> idx = flattenMatrixSampleIndex(alpha[len-2], cd);
        sample[len-2] = idx.a;
        sample[len-1] = idx.b;
        for (int i = len-3; i >= 0; i--) {
            sample[i] = cd.sample(stSlice(trigamma[i+1], sample[i+1], sample[i+2]));
        }
        return sample;
    }

    public void parameterSampling(int[] hiddenSample) {
        for (Counters counter : empiricalCounts) {
            counter.resetAll();
        }
        for (int t = 0; t < len; t++) {
            empiricalCounts[hiddenSample[t]].increment(obs[t]);
        }
        for (int i = 0; i < nt; i++) {
            B[i] = dirichet.sample(empiricalCounts[i]);
        }
    }

    public void setSeed(long seed){
        rng.setSeed(seed);
    }

    /*
    main training procedure
     */
    public void gibbsSampling(int nIter, boolean startWithParameter, int[] firstSample, long seed, boolean verbose) {
        setSeed(seed);
        if (startWithParameter){
            init();
        } else if (firstSample != null) {
            parameterSampling(firstSample);
        } else {
            firstSample = firstSample();
        }
        logProb = new double[nIter];
        long start;
        long stop;
        for (int it = 0; it < nIter; it++) {
            if (verbose) System.out.printf("iteration: %d\n", it);
            start = System.nanoTime();
            forward();
            int[] hiddenSample = backwardSampling();
            parameterSampling(hiddenSample);
            logProb[it] = logProb(hiddenSample);
            stop = System.nanoTime();
            if (verbose) {
                System.out.println(logProb[it]);
                System.out.println(Arrays.toString(hiddenSample));
                System.out.printf("time: %f\n", (stop - start) / 1e9);
            }
        }
    }



    public void init() {
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

    /*
    compute logProb of the hidden obs based on trigram only
     */
    public double logProb(int[] sample){
        double lp = 0;
        for (int i = 2; i < len; i++) {
            lp += FastMath.log(A[sample[i-2]][sample[i-1]][sample[i]]);
        }
        return lp;
    }

    /*
    Randomly create a first key
     */
    public int[] firstSample() {
        String mostCommon = "etaoinshrdlcumwfgypbvkjxqz";
        HashMap<Integer, LinkedList<Integer>> dict = new HashMap<>();
        for (int i = 0; i < len; i++) {
            if (!dict.containsKey(obs[i])) {
                dict.put(obs[i], new LinkedList<Integer>());
            }
            dict.get(obs[i]).add(i);
        }
        List<Map.Entry<Integer, LinkedList<Integer>>> list = new ArrayList(dict.entrySet());
        Collections.sort(list, (a,b) -> b.getValue().size() - a.getValue().size()); // descending

        int[] sample = new int[len];
        int rank = 0;
        for (Map.Entry<Integer, LinkedList<Integer>> entry : list) {
            int o = entry.getKey();
            int x = mostCommon.charAt(rank % mostCommon.length()) - 97;
            assert(x < 26);
            for (int idx : entry.getValue()) {
                sample[idx] = x;
            }
            rank += 1;
        }
        return sample;
    }

    public int[] viterbi() {
        double c = 0;

        int[][][] dp = new int[len - 1][nt][nt];
        double[][] head = new double[nt][nt];
        double[][] temp = new double[nt][nt];

        // i -> j
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                head[i][j] = alpha[0][i][j];
            }
        }

        for (int t = 0; t < len - 2; t++) {
            c = 0;
            // i -> (j -> k), fix j and k, search over i
            for (int k = 0; k < nt; k++) {
                for (int j = 0; j < nt; j++) {
                    double x = 0;
                    double max = Double.NEGATIVE_INFINITY;
                    int argmax = -1;
                    for (int i = 0; i < nt; i++) {
                        x = head[i][j] * A[i][j][k];
                        if (x > max) {
                            max = x;
                            argmax = i;
                        }
                    }
                    dp[t][j][k] = argmax;
                    temp[j][k] = max * Math.pow(B[k][obs[t + 2]], 3);
                    c += temp[j][k];
                }
            }
            //scale and move back to head
            c = 1 / c;
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    head[i][j] = c * temp[i][j];
                }
            }
        }

        //get the last state of the best DP sequence
        int argmaxi = -1;
        int argmaxj = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                if (head[i][j] > max) {
                    max = head[i][j];
                    argmaxi = i;
                    argmaxj = j;
                }
            }
        }
        //trace back
        int[] resn = new int[len];
        resn[len - 2] = argmaxi;
        resn[len - 1] = argmaxj;
        for (int t = len - 3; t >= 0; t--) {
            argmaxi = dp[t][argmaxi][argmaxj];
            argmaxj = argmaxi;
            resn[t] = argmaxi;
        }
        char[] resc = new char[len];
        for (int i = 0; i < len; i++) {
            resc[i] = (char) (resn[i] + 65);
        }
        return resn;
    }


    static double[] stSlice(double[][][] a, int nd, int rd){
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i][nd][rd];
        }
        return res;
    }

    static Pair<Integer, Integer> flattenMatrixSampleIndex(double[][] matrix, CategoricalDistribution cd){
        int nrows = matrix.length;
        int ncols = matrix[0].length;
        double[] flat = new double[nrows*ncols];
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                flat[i*ncols+j] = matrix[i][j];
            }
        }
        int idx = cd.sample(flat);
        return new Pair<Integer, Integer>(idx / ncols, idx % ncols);
    }

    static class Pair<A,B> {
        A a;
        B b;
        public Pair(A a, B b){
            this.a = a;
            this.b = b;
        }
    }

    public static void main(String[] args) throws Exception {
        int[] pt = util.plain("data/408plaincleaned");
        int[] ct = util.read408("data/408ciphercleaned");
        int nTag = 26;
        int nWord = 54;
        int len = 408;

        double[][][] trigram = util.readTrigram("data/trigram.txt");
        long start = System.nanoTime();
        HMMBlockSampler sampler = new HMMBlockSampler(trigram, nTag, nWord, ct);
        long stop = System.nanoTime();
        sampler.gibbsSampling(200, false, null, -3076155353333121539L, true);
//        FHMM.train(200, false, 8781939572407739913L, true);
//        FHMM.train(200, false, 1209845257843231593L, true);
//        FHMM.train(200, false, 3738420990656387694L, true);

        stop = System.nanoTime();
        System.out.printf("training time: %f\n", (stop - start) / 1e9);

        /*
        Use Viterbi decode
         */
//        int[] sol = sampler.viterbi();
//        double acc = 0;
//        for (int i = 0; i < sol.length; i++) {
//            if (sol[i] == pt[i]) {
//                acc += 1;
//            }
//        }
//        System.out.printf("accuracy = %f\n", acc / len);
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < sol.length; i++) {
//            sb.append(Character.toString((char) (sol[i] + 65)));
//        }
//        System.out.println(sb.toString());
//        System.out.println(util.display2(FHMM.B));
//        BufferedWriter out = new BufferedWriter(new FileWriter("B_1236"));
//        for (int i = 0; i < 26; i++) {
//            for (int j = 0; j < 54; j++) {
//                out.write(String.format("%f ", FHMM.B[i][j]));
//            }
//            out.newLine();
//        }
//        out.close();

        /*
        Use highest-by-column decode
         */
//        int[] argmax = new int[FHMM.nw];
//                for (int k = 0; k < FHMM.nw; k++) {
//                    double max = FHMM.B[0][k];
//                    int arg = 0;
//                    for (int l = 1; l < FHMM.nt; l++) {
//                        if (max < FHMM.B[l][k]) {
//                            max = FHMM.B[l][k];
//                            arg = l;
//                        }
//                    }
//                    argmax[k] = arg;
//                }
//                int[] sol_num = new int[len];
//                double acc = 0;
//                for (int k = 0; k < len; k++) {
//                    sol_num[k] = argmax[ct[k]];
//                    if (sol_num[k] == pt[k]) {
//                        acc++;
//                    }
//                }
//                acc /= len;
//        System.out.println(acc);
    }
}
