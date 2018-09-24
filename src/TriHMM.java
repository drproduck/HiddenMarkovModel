import org.apache.commons.math3.util.FastMath;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;

public class TriHMM {
    double[][][] A;
    double[][] B;
    int len;
    int nt;
    int nw;
    double[][][] ap, bt;
    double[][][] digamma;
    double[][][][] trigamma;
    double[][] gamma;
    int[] seq;
    double logProb;
    double oldLogProb;
    boolean firstRestart;
    static double eps = FastMath.log(0.1);

    public TriHMM(double[][][] transition, int nTag, int nWord, int[] sequence) {
        len = sequence.length;
        nt = nTag;
        nw = nWord;
        A = transition;
        B = new double[nt][nw];
        ap = new double[len][nt][nt];
        bt = new double[len][nt][nt];
        digamma = new double[len][nt][nt];
        trigamma = new double[len][nt][nt][nt];
        gamma = new double[len][nt];
        seq = sequence;
        ap = new double[len - 1][nt][nt];
        bt = new double[len - 1][nt][nt];
        firstRestart = true;
    }

    public void alpha() {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                ap[0][i][j] = B[i][seq[0]] + B[j][seq[1]];
            }
        }
        for (int t = 1; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    double[] acc = new double[nt];
                    for (int k = 0; k < nt; k++) {
                        acc[k] = ap[t - 1][k][i] + A[k][i][j] + B[j][seq[t + 1]];
                    }
                    ap[t][i][j] = logSumExp(acc);
                }
            }
        }
    }

    public double logProbFromAlpha() {
        double[] acc = new double[nt];
        for (int i = 0; i < nt; i++) {
            acc[i] = logSumExp(ap[len - 2][i]);
        }
        logProb = logSumExp(acc);
        return logProb;
    }

    public void beta() {
        // nominal code
//        for (int i = 0; i < nt; i++) {
//            for (int j = 0; j < nt; j++) {
//                beta[len-2][i][j] = 0;
//            }
//        }
        for (int t = len-3; t >= 0; t--) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    double[] acc = new double[nt];
                    for (int k = 0; k < nt; k++) {
                        acc[k] = bt[t + 1][j][k] + A[i][j][k] + B[k][seq[t + 2]];
                    }
                    bt[t][i][j] = logSumExp(acc);
                }
            }
        }
    }

    public void digamma() {
        for (int t = 0; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    digamma[t][i][j] = ap[t][i][j] + bt[t][i][j] - logProb;
                }
            }
        }
    }

    public void trigamma() {
        for (int t = 0; t < len - 2; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        trigamma[t][i][j][k] = ap[t][i][j] + A[i][j][k] + B[k][seq[t + 2]] + bt[t + 1][j][k] - logProb;
                    }
                }
            }
        }
    }

    public void gamma() {
        for (int t = 0; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                gamma[t][i] = logSumExp(digamma[t][i]);
            }
        }

        for (int j = 0; j < nt; j++) {
            double[] acc = new double[nt];
            for (int i = 0; i < nt; i++) {
                acc[i] = digamma[len - 2][i][j];
            }
            gamma[len - 1][j] = logSumExp(acc);
        }
    }

    public void reEstimate(boolean trainA) {
        if (firstRestart) {
            alpha(); logProbFromAlpha();
            oldLogProb = logProb;
            firstRestart = false;
        }
        beta(); digamma(); gamma();
        if (trainA) {
            trigamma();
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        double[] arr = new double[len - 2];
                        for (int t = 0; t < len - 2; t++) {
                            arr[t] = trigamma[t][i][j][k];
                        }

                        A[i][j][k] = logSumExp(arr);
                    }
                    logSumExpForSoftMax(A[i][j]);
                }
            }
        }
        double[][] BB = new double[nt][nw];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nw; j++) {
                List<Double> arr = new ArrayList<>();
                for (int t = 0; t < len; t++) {
                    if (seq[t] == j) {
                        arr.add(gamma[t][i]);
                    }
                }
                arr.add(eps);
                BB[i][j] = logSumExp(arr);
            }
            logSumExpForSoftMax(BB[i]);
        }
        B = BB;
        alpha(); logProbFromAlpha();
    }

    public void train(int iter, boolean trainA, long seed, boolean verbose) {
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

    public static double logSumExp(double[] arr) {
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            max = (max < arr[i]) ? arr[i] : max;
        }
        double s = 0;
        for (int i = 0; i < arr.length; i++) {
            s += FastMath.exp(arr[i] - max);
        }
        return max + FastMath.log(s);
    }

    public static double logSumExp(List<Double> arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (Double d : arr) {
            max = (max < d) ? d : max;
        }
        double s = 0;
        for (Double d : arr) {
            s += FastMath.exp(d - max);
        }
        if (Double.isInfinite(s)) {
            System.out.println("infinity error");
        }
        return max + FastMath.log(s);
    }

    public static void logSumExpForSoftMax(double[] inPlaceArr) {
        double max = inPlaceArr[0];
        for (int i = 0; i < inPlaceArr.length; i++) {
            max = (max < inPlaceArr[i]) ? inPlaceArr[i] : max;
        }
        double s = 0;
        for (int i = 0; i < inPlaceArr.length; i++) {
            s += FastMath.exp(inPlaceArr[i] - max);
        }
        s = max + FastMath.log(s);
        for (int i = 0; i < inPlaceArr.length; i++) {
            inPlaceArr[i] = inPlaceArr[i] - s;
        }
    }

    public static void logSumExpForSoftMax(List<Double> inPlaceArr) {
        double max = Double.NEGATIVE_INFINITY;
        for (Double d : inPlaceArr) {
            max = (max < d) ? d : max;
        }
        double s = 0;
        for (Double d : inPlaceArr) {
            s += FastMath.exp(d - max);
        }
        s = max + FastMath.log(s);
        for (int i = 0; i < inPlaceArr.size(); i++) {
            inPlaceArr.set(i, inPlaceArr.get(i) - s);
        }
    }

    public void init(long seed, boolean initA) {
        Random rng = new Random(seed);
        double s;
        double c;
        for (int i = 0; i < nt; i++) {
            s = 0;
            for (int j = 0; j < nw; j++) {
                B[i][j] = rng.nextDouble();
                s += B[i][j];
            }
            s = FastMath.log(s);
            for (int j = 0; j < nw; j++) {
                B[i][j] = FastMath.log(B[i][j]) - s;
            }
        }
        if (initA) {
            A = new double[nt][nt][nt];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    s = 0;
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] = rng.nextDouble();
                        s += A[i][j][k];

                    }
                    s = FastMath.log(s);
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] = FastMath.log(A[i][j][k]) - s;
                    }
                }
            }
        }
    }

//    public int[] maxExpectation() {
//
//    }

    public int[] viterbi() {
        int[][][] dp = new int[len - 1][nt][nt];
        double[][] head = new double[nt][nt];
        double[][] temp = new double[nt][nt];

        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                head[i][j] = 3 * B[i][seq[0]] + 3 * B[j][seq[1]];
            }
        }

        // in order i -> j -> k
        for (int t = 0; t < len - 2; t++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    double x;
                    double max = Double.NEGATIVE_INFINITY;
                    int argmax = -1;
                    for (int i = 0; i < nt; i++) {
                        x = head[i][j] + A[i][j][k];
                        if (x > max) {
                            max = x;
                            argmax = i;
                        }
                    }
                    dp[t][j][k] = argmax;
                    temp[j][k] = max + 3 * B[k][seq[t + 2]];
                }
            }
            head = util.deepClone(temp);

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

    public static void main(String[] args) throws Exception {
//        double[] arr = {-750.7997981153557, -747.2419398916022, -749.5203135833029, -749.4738954519979, -748.9035294593572, -748.4535571990109};
//        List<Double> a = new ArrayList<>();
//        for (int i = 0; i < arr.length; i++) {
//            a.add(arr[i]);
//        }
//        System.out.println(logSumExp(a));


        int[] pt = util.plain("/home/drproduck/Documents/HMM/data/408plaincleaned");
        int[] ct = util.read408("/home/drproduck/Documents/HMM/data/408ciphercleaned");
        int nTag = 26;
        int nWord = 54;
        int len = 408;

        double[][][] trigram = util.readTrigram("/home/drproduck/Documents/HMM/data/logtrigram.txt");
        long start = System.nanoTime();
        TriHMM HMM = new TriHMM(trigram, nTag, nWord, ct);
        long stop = System.nanoTime();
        System.out.printf("initialization time: %f\n", (stop - start) / 1e9);

        start = System.nanoTime();
        HMM.train(100, false, -3076155353333121539L, true);
//        HMM.train(200, false, 8781939572407739913L, true);
//        HMM.train(200, false, 1209845257843231593L, true);
//        HMM.train(100, false, 1236, true);
        stop = System.nanoTime();
        System.out.printf("training time: %f\n", (stop - start) / 1e9);
        int[] sol = HMM.viterbi();
        double acc = 0;
        for (int i = 0; i < sol.length; i++) {
            if (sol[i] == pt[i]) {
                acc += 1;
            }
        }
        System.out.printf("accuracy = %f\n", acc / len);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sol.length; i++) {
            sb.append(Character.toString((char) (sol[i] + 65)));
        }
        System.out.println(sb.toString());
        System.out.println(util.display2(HMM.B));
        BufferedWriter out = new BufferedWriter(new FileWriter("B_1236"));
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 54; j++) {
                out.write(String.format("%f ", HMM.B[i][j]));
            }
            out.newLine();
        }
        out.close();

        // Use this if you want the highest-by-column key derivation.
//        int[] argmax = new int[HMM.nw];
//                for (int k = 0; k < HMM.nw; k++) {
//                    double max = HMM.B[0][k];
//                    int arg = 0;
//                    for (int l = 1; l < HMM.nt; l++) {
//                        if (max < HMM.B[l][k]) {
//                            max = HMM.B[l][k];
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
