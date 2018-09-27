import org.apache.commons.math3.util.FastMath;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 * Trigram HMM. a similar normalization is used instead of log-scale and the ensuing logSumExp.
 */

public class QuadHMM {

    double[][][][] A;
    double[][] B;
    int len;
    int nt;
    int nw;
    double[][][][] ap, bt;
    double[] c;

    double[][][] digamma;
    double[][][][] trigamma;
    double[][] gamma;
    int[] seq;
    double logProb;
    double oldLogProb;
    boolean firstRestart;
    static double eps = 0.1; // smoothing constant

    public QuadHMM(double[][][][] transition, int nTag, int nWord, int[] sequence) {
        len = sequence.length;
        nt = nTag;
        nw = nWord;
        A = transition;
        B = new double[nt][nw];
        ap = new double[len - 1][nt][nt][nt];
        bt = new double[len - 1][nt][nt][nt];
        c = new double[len - 1];

        trigamma = new double[len][nt][nt][nt];
        gamma = new double[len][nt];
        seq = sequence;

        firstRestart = true;
    }

    /*
    p(x_t, x_{t+1}, x_{t+2} | O_0^{t+2})
    c[0] = p(O_0, O_1, O_2)
    c[t] = p(O_{t+2} | O_0^{t+1})
    t = 0 -> len - 1 - 2
     */
    public void alpha() {
        c[0] = 0;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    ap[0][i][j][k] = B[i][seq[0]] * B[j][seq[1]] * B[k][seq[2]];
                    c[0] += ap[0][i][j][k];
                }
            }
        }
        c[0] = 1 / c[0];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    ap[0][i][j][k] *= c[0];
                }
            }
        }
        for (int t = 1; t < len - 2; t++) {
            c[t] = 0;
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    for (int l = 0; l < nt; l++) {
                        ap[t][j][k][l] = 0;
                        for (int i = 0; i < nt; i++) {
                            ap[t][j][k][l] += ap[t - 1][i][j][k] * A[i][j][k][l] * B[l][seq[t + 2]];
                        }
                        c[t] += ap[t][j][k][l];
                    }
                }
            }
            c[t] = 1 / c[t];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        ap[t][i][j][k] *= c[t];
                    }
                }
            }
        }
    }

    public double logProbFromAlpha() {
        logProb = 0;
        for (int i = 0; i < len - 2; i++) {
            logProb -= FastMath.log(c[i]);
        }
        return logProb;
    }


    /*

    bigram: recursion on p(O_t^n|x_{t-1})
    p(O^n|x_{n-1}) = \sum_{x_n} p(O_n|x_n) p(x_n|x_{n-1})
    p(O_t^n|x_{t-1}) = \sum_{x_t} p(O_t, O_{t+1}^n, x_t|x_{t-1}) = \sum_{x_t} p(O_t|x_t) p(O_{t+1}^n|x_t) p(x_t|x_{t-1})

    p(O_t^n|x_{t-1}) / p(O_t|O_0^{t-1})

    quadgram: recursion on p(O_t^n|x_{t-1},x_{t-2},x_{t-3})
    p(O_t^n|x_{t-1},x_{t-2},x_{t-3}) = \sum_{x_t} p(O^t|x_t) p(O_{t+1}^n|x_t,x_{t-1},x_{t-2}) p(x_t|x_{t-1},x_{t-2},x_{t-3})
     */

    /*
    p(x_t, x_{t+1}, x_{t+2} | O_0^{t+2}) * p(O_{t+3}^n | x_t, x_{t+1}, x_{t+2}) = p(O_{t+3}^n, x_t, x_{t+1}, x_{t+2} | O_0^{t+2})
    need to divide by p(O_{t+3}^n | O_0^{t+2}) = p(O_{t+3}|O_0^{t+2}) * p(O_{t+4}|O_0^{t+3}) * ... * p(O_n|O_0^{n-1})
     */

    public void beta() {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    bt[len - 3][i][j][k] = 1; // p(O_n,O_{n-1}|O_0^{n-2})
                }
            }
        }
        for (int t = len - 4; t >= 0; t--) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        bt[t][i][j][k] = 0;
                        for (int l = 0; l < nt; l++) {
                            bt[t][i][j][k] += bt[t + 1][j][k][l] * A[i][j][k][l] * B[l][seq[t + 3]];
                        }
                        bt[t][i][j][k] *= c[t+1];
                    }
                }
            }
        }
    }

    /*
    p(x_t,x_{t+1},x_{t+2}|O)
     */
    public void marginalTrigram(){
        for (int t = 0; t < len - 2; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        trigamma[t][i][j][k] = ap[t][i][j][k] * bt[t][i][j][k];
                    }
                }
            }
        }
    }

    public void marginalUnigram(){
        for (int t = 0; t < len; t++) {
            for (int i = 0; i < nt; i++) {
                gamma[t][i] = 0;
            }
        }
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    gamma[0][i] += trigamma[0][i][j][k];
                    gamma[1][j] += trigamma[0][i][j][k];
                    gamma[2][k] += trigamma[0][i][j][k];
                }
            }
        }
        for (int t = 1; t < len - 2; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        gamma[t+2][k] += trigamma[t][i][j][k];
                    }
                }
            }
        }
    }



    public void reEstimate() {
        if (firstRestart) {
            alpha();
            logProbFromAlpha();
            oldLogProb = logProb;
            firstRestart = false;
        }
        beta();
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nw; j++) {
                double denom = eps * nw;
                double numer = 0;
                for (int t = 0; t < len; t++) {
                    if (seq[t] == j) {
                        numer += gamma[t][i];
                    }
                    denom += gamma[t][i];
                }
                numer += eps;
                B[i][j] = numer / denom;
            }
        }
        alpha();
        logProbFromAlpha();
    }

    public void train(int iter, long seed, boolean verbose) {
        init(seed);
        long start;
        long stop;
        for (int it = 0; it < iter; it++) {
            if (verbose) System.out.printf("iteration: %d\n", it);
            start = System.nanoTime();
            reEstimate();
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

    public void init(long seed) {
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
    }

    public int[] viterbi() {
        double c = 0;

        int[][][][] dp = new int[len - 1][nt][nt][nt];
        double[][][] head = new double[nt][nt][nt];
        double[][][] temp = new double[nt][nt][nt];

        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    head[i][j][k] = ap[0][i][j][k];
                }
            }
        }

        for (int t = 0; t < len - 3; t++) {
            c = 0;
            // i -> (j -> k -> l), fix j, k and l, search over i
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    for (int l = 0; l < nt; l++) {
                        double x = 0;
                        double max = Double.NEGATIVE_INFINITY;
                        int argmax = -1;
                        for (int i = 0; i < nt; i++) {
                            x = head[i][j][k] * A[i][j][k][l];
                            if (x > max) {
                                max = x;
                                argmax = i;
                            }
                        }
                        dp[t][j][k][l] = argmax;
                        temp[j][k][l] = max * Math.pow(B[l][seq[t + 3]], 3);
                        c += temp[j][k][l];
                    }
                }
            }
            //scale and move back to head
            c = 1 / c;
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        head[i][j][k] = c * temp[i][j][k];
                    }
                }
            }
        }

        //get the last state of the best DP sequence
        int argmaxi = -1;
        int argmaxj = -1;
        int argmaxk = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                for (int k = 0; k < nt; k++) {
                    if (head[i][j][k] > max) {
                        max = head[i][j][k];
                        argmaxi = i;
                        argmaxj = j;
                        argmaxk = k;
                    }
                }
            }
        }
        //trace back
        int[] resn = new int[len];
        resn[len - 3] = argmaxi;
        resn[len - 2] = argmaxj;
        resn[len - 1] = argmaxk;

        for (int t = len - 4; t >= 0; t--) {
            argmaxi = dp[t][argmaxi][argmaxj][argmaxk];
            argmaxj = argmaxi;
            argmaxk = argmaxj;

            resn[t] = argmaxi;
        }
        char[] resc = new char[len];
        for (int i = 0; i < len; i++) {
            resc[i] = (char) (resn[i] + 65);
        }
        return resn;
    }

    public static void main(String[] args) throws Exception {
//        int[] pt = util.plain("data/408plaincleaned");
//        int[] ct = util.read408("data/408ciphercleaned");
//        int nTag = 26;
//        int nWord = 54;
//        int T = 408;

        int nTag = 26;
        int nWord = 26;
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
        double[][][][] quadgram = readQuadgram("data/quadgram");
        long start = System.nanoTime();
        QuadHMM FHMM = new QuadHMM(quadgram, nTag, nWord, ct);
        long stop = System.nanoTime();
        FHMM.train(5,-3076155353333121539L, true);
//        FHMM.train(200, false, 8781939572407739913L, true);
//        FHMM.train(200, false, 1209845257843231593L, true);
//        FHMM.train(200, false, 3738420990656387694L, true);

        stop = System.nanoTime();
        System.out.printf("training time: %f\n", (stop - start) / 1e9);

        /*
        Use Viterbi decode
         */
        int[] sol = FHMM.viterbi();
        double acc = 0;
        for (int i = 0; i < sol.length; i++) {
            if (sol[i] == pt[i]) {
                acc += 1;
            }
        }
        System.out.printf("accuracy = %f\n", acc / T);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sol.length; i++) {
            sb.append(Character.toString((char) (sol[i] + 65)));
        }
        System.out.println(sb.toString());
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

    public static double[][][][] readQuadgram(String dir) throws FileNotFoundException {
        Scanner rd = new Scanner(new BufferedReader(new FileReader(dir)));
        double[][][][] quad = new double[26][26][26][26];
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                for (int k = 0; k < 26; k++) {
                    for (int l = 0; l < 26; l++) {
                        quad[i][j][k][k] = rd.nextDouble();
                    }
                }
            }
        }
        return quad;
    }
}
