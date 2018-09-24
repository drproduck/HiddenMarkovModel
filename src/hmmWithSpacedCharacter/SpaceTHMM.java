package hmmWithSpacedCharacter;

import org.apache.commons.math3.util.FastMath;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 * Trigram HMM. a similar normalization is used instead of log-scale and the ensuing logSumExp.
 * for each character c, a new character _c is added to accomodate for the space character
 */

public class SpaceTHMM {

    double[][][] A;
    double[][] B;
    int len;
    int nChar;
    int nt;
    int nw;
    double[][][] ap, bt;
    double[] c;
    double[][][] digamma;
    double[][][][] trigamma;
    double[][] gamma;
    int[] seq;
    double logProb;
    double oldLogProb;
    boolean firstRestart;
    static double eps = 0.1; // smoothing constant

    public SpaceTHMM(double[][][] transition, int nTag, int nWord, int[] sequence) {
        len = sequence.length;
        nChar = nTag;
        nt = nTag*2;
        nw = nWord;
        A = transition;
        B = new double[nt][nw];
        ap = new double[len - 1][nt][nt];
        bt = new double[len - 1][nt][nt];
        c = new double[len - 1];
        digamma = new double[len][nt][nt];
        trigamma = new double[len][nt][nt][nt];
        gamma = new double[len][nt];
        seq = sequence;

        firstRestart = true;
    }

    /*
    To share B parameter, convert _c to c
     */
    public int toChar(int c) {
        return (c > nChar) ? c - nChar : c;
    }

    public void alpha() {
        c[0] = 0;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                ap[0][i][j] = B[toChar(i)][seq[0]] * B[toChar(j)][seq[1]];
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
                        ap[t][i][j] += ap[t - 1][k][i] * A[k][i][j] * B[toChar(j)][seq[t + 1]];
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

    public double logProbFromAlpha() {
        logProb = 0;
        for (int i = 0; i < len - 1; i++) {
            logProb -= FastMath.log(c[i]);
        }
        return logProb;
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
                        bt[t][i][j] += bt[t + 1][j][k] * A[i][j][k] * B[toChar(k)][seq[t + 2]];
                    }
                    bt[t][i][j] *= c[t + 1]; // p(O2|x0,x1) / p(O2|O0,O1)
                }
            }
        }
    }

    public void digamma() {
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
    }

    public void trigamma() {
        for (int t = 0; t < len - 2; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        trigamma[t][i][j][k] = ap[t][i][j] * A[i][j][k] * B[toChar(k)][seq[t + 2]] * bt[t + 1][j][k] * c[t + 1];
                        // (p(O0,O1,x0,x1)/p(O0,O1)) * p(x2|x0,x1) * p(O2|x2) * (p(O3|x1,x2)/p(O3|O1,O2)) / p(O2|O0,O1)
                    }
                }
            }
        }
    }

    public void reEstimate(boolean trainA) {
        if (firstRestart) {
            alpha();
            logProbFromAlpha();
            oldLogProb = logProb;
            firstRestart = false;
        }
        beta();
        digamma();
        if (trainA) {
            trigamma();
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    double denom = 0;
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] = 0;
                        for (int t = 0; t < len - 2; t++) {
                            A[i][j][k] += trigamma[t][i][j][k];
                        }
                        denom += A[i][j][k];
                    }
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] /= denom;
                    }
                }
            }
        }

        /*
        update B.
         */
        for (int i = 0; i < nChar; i++) {
            for (int j = 0; j < nw; j++) {
                double denom = eps * nw;
                double numer = 0;
                for (int t = 0; t < len; t++) {
                    if (seq[t] == j) {
                        numer += gamma[t][i] + gamma[t][i+nChar]; // share parameter
                    }
                    denom += gamma[t][i] + gamma[t][i+nChar];
                }
                numer += eps;
                B[i][j] = numer / denom;
            }
        }
        alpha();
        logProbFromAlpha();
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

    public void init(long seed, boolean initA) {
        Random rng = new Random(seed);
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
        if (initA) {
            A = new double[nt][nt][nt];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    s = 0;
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] = rng.nextDouble();
                        s += A[i][j][k];

                    }
                    for (int k = 0; k < nt; k++) {
                        A[i][j][k] /= s;
                    }
                }
            }
        }
    }

    public int[] viterbi() {
        double c = 0;

        int[][][] dp = new int[len - 1][nt][nt];
        double[][] head = new double[nt][nt];
        double[][] temp = new double[nt][nt];

        // i -> j
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                head[i][j] = ap[0][i][j];
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
                    temp[j][k] = max * Math.pow(B[toChar(k)][seq[t + 2]], 3);
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

    public static void main(String[] args) throws Exception {
//        int[] pt = utilSpace.plain("data/408plaincleaned");
//        int[] ct = utilSpace.read408("data/408ciphercleaned");
//        int nTag = 26;
//        int nWord = 54;
//        int T = 408;
        /*
        simple cipher
         */
        int nTag = 26;
        int nWord = 26;
        int maxIters = 100;
        int T = 500;
        String cipherDir = "simple_cipher/simple_cipher_length=500";
        Scanner reader = new Scanner(new BufferedReader(new FileReader(cipherDir)));
        int[] pt = new int[T];
        for (int i = 0; i < T; i++) {
            pt[i] = reader.nextInt();
        }
        int[] ct = new int[T];
        for (int i = 0; i < T; i++) {
            ct[i] = reader.nextInt();
        }

        double[][][] trigram = utilSpace.trigram(Integer.MAX_VALUE, 0, true);
        System.out.println(Arrays.stream(trigram[0][0]).sum());
        long start = System.nanoTime();
        SpaceTHMM FHMM = new SpaceTHMM(trigram, nTag, nWord, ct);
        long stop = System.nanoTime();
//        FHMM.train(200, false, -3076155353333121539L, true);
//        FHMM.train(100, false, 8781939572407739913L, true);
        FHMM.train(200, false, 1209845257843231593L, true);
//        FHMM.train(200, false, 3738420990656387694L, true);

        stop = System.nanoTime();
        System.out.printf("training time: %f\n", (stop - start) / 1e9);

        /*
        Use Viterbi decode
         */
        int[] sol = FHMM.viterbi();
        double acc = 0;
        for (int i = 0; i < sol.length; i++) {
            if (FHMM.toChar(sol[i]) == pt[i]) {
                acc += 1;
            }
        }
        System.out.printf("accuracy = %f\n", acc / T);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sol.length; i++) {
            if (sol[i] < nTag) {
                sb.append(Character.toString((char) (sol[i] + 65)));
            } else {
                sb.append("_");
                sb.append(Character.toString((char) (sol[i] - nTag + 65)));
            }

        }
        System.out.println(sb.toString());

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
//        for (int k = 0; k < FHMM.nw; k++) {
//            double max = FHMM.B[0][k];
//            int arg = 0;
//            for (int l = 1; l < FHMM.nt; l++) {
//                if (max < FHMM.B[l][k]) {
//                    max = FHMM.B[l][k];
//                    arg = l;
//                }
//            }
//            argmax[k] = arg;
//        }
//        int[] sol = new int[T];
//        double acc = 0;
//        for (int k = 0; k < T; k++) {
//            sol[k] = argmax[ct[k]];
//            if (sol[k] == pt[k]) {
//                acc++;
//            }
//        }
//        acc /= T;
//        System.out.println(acc);
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < sol.length; i++) {
//            if (sol[i] < nTag) {
//                sb.append(Character.toString((char) (sol[i] + 65)));
//            } else {
//                sb.append("_");
//                sb.append(Character.toString((char) (sol[i] - nTag + 65)));
//            }
//
//        }
//        System.out.println(sb.toString());
    }
}
