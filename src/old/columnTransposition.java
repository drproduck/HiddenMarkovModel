import em.TriHMM;
import em.map.MapHMM;
import em.map.MDir;
import util.Helper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class columnTransposition {
    static Random rng = new Random(12345);

    public static void main(String[] args) throws Exception{
//        int[] swapPair = {0, 1};
//        while (swapPair != null) {
//            System.out.println(Arrays.toString(swapPair));
//            swapPair = nextPair(swapPair);
//        }
//        old.makeA(2, 100000, 200, 10, 1234, "/home/drproduck/Documents/em.HMM/data/triA2_restarts");
//        columnTransposeWithTrigram();
//        columnTransposeWithBigram();
        int[] col =      {2, 10, 3, 6, 0, 9, 5, 11, 4, 1, 7, 12, 8, 13, 14, 15, 16};
        int[] truePerm = {11, 15, 8, 4, 12, 3, 9, 10, 5, 13, 1, 2, 0, 7, 6, 14, 16};
        int[] generatedPerm = new int[17];
        for (int i = 0; i < 17; i++) {
            generatedPerm[i] = truePerm[col[i]];
        }
        System.out.println(Arrays.toString(col));
        System.out.println(Arrays.toString(generatedPerm));

    }

    public static void columnTransposeWithBigram() throws Exception{
        int dim = 4;
        Scanner in = new Scanner(new BufferedReader(new FileReader("/home/drproduck/Documents/em.HMM/data/biA4")));
        double[][] A = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                A[i][j] = in.nextDouble();
            }
        }
        int[][] baseText = readCipherMatrix("/home/drproduck/Documents/em.HMM/data/408plaintransposed");
        int[] col = new int[17];
        for (int i = 0; i < 17; i++) {
            col[i] = i;
        }
        int[] candidate;
        double[] alpha = new double[26];
        for (int i = 0; i < 26; i++) {
            alpha[i] = 1.01;
        }
        MDir prior = new MDir(26, alpha, 0.0001);
        MapHMM HMM = new MapHMM(dim, 26, flattenMatrix(baseText, col), prior);
        HMM.A = A;
        double bestLogProb = testLogProb(HMM, flattenMatrix(baseText, col), dim, 200, 10);
        double prob;
        int[] swapPair = {0, 1};
        while (swapPair != null) {
            candidate = swapCol(swapPair, col);
            prob = testLogProb(HMM, flattenMatrix(baseText, candidate), dim, 200, 100);
            if (prob > bestLogProb) {
                System.out.printf("SWAPPED, new logProb = %f, swap pair = %d %d\n", prob, swapPair[0], swapPair[1]);
                bestLogProb = prob;
                col = candidate;
                swapPair = new int[]{0, 1};
            } else {
                swapPair = nextPair(swapPair);
            }
        }
        int[] truePerm = {11, 15, 8, 4, 12, 3, 9, 10, 5, 13, 1, 2, 0, 7, 6, 14, 16};
        int[] generatedPerm = new int[17];
        for (int i = 0; i < 17; i++) {
            generatedPerm[i] = truePerm[col[i]];
        }
        System.out.println(Arrays.toString(col));
        System.out.println(Arrays.toString(generatedPerm));
    }

    public static void columnTransposeWithTrigram() throws Exception{
        int dim = 2;
        Scanner in = new Scanner(new BufferedReader(new FileReader("/home/drproduck/Documents/em.HMM/data/triA2")));
        double[][][] A = new double[dim][dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    A[i][j][k] = in.nextDouble();
                }
            }
        }
        int[][] baseText = readCipherMatrix("/home/drproduck/Documents/em.HMM/data/408plaintransposed");
        int[] col = new int[17];
        for (int i = 0; i < 17; i++) {
            col[i] = i;
        }
        int[] candidate;
        TriHMM HMM = new TriHMM(A, dim, 26, flattenMatrix(baseText, col));
        double bestLogProb = testLogProb(HMM, flattenMatrix(baseText, col), dim, 100, 100);
        double prob;
        int[] swapPair = {0, 1};
        while (swapPair != null) {
            candidate = swapCol(swapPair, col);
            prob = testLogProb(HMM, flattenMatrix(baseText, candidate), dim, 100, 100);
            if (prob > bestLogProb) {
                System.out.printf("SWAPPED, new logProb = %f, swap pair = %d %d\n", prob, swapPair[0], swapPair[1]);
                bestLogProb = prob;
                col = candidate;
                swapPair = new int[]{0, 1};
            } else {
                swapPair = nextPair(swapPair);
            }
        }
        int[] truePerm = {11, 15, 8, 4, 12, 3, 9, 10, 5, 13, 1, 2, 0, 7, 6, 14, 16};
        int[] generatedPerm = new int[17];
        for (int i = 0; i < 17; i++) {
            generatedPerm[i] = truePerm[col[i ]];
        }
        System.out.println(Arrays.toString(col));
        System.out.println(Arrays.toString(generatedPerm));
    }

    public static double testLogProb(TriHMM HMM, int[] seq, int dim, int iter, int restart) {
        double prob;
        double bestLogProb = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < restart; i++) {
            HMM.firstRestart = true;
            HMM.seq = seq;
            HMM.train(iter, true, rng.nextInt(), false);
            prob = HMM.logProb;
            if (prob > bestLogProb) {
                bestLogProb = prob;
            }
        }
        return bestLogProb;
    }

    public static double testLogProb(MapHMM HMM, int[] seq, int dim, int iter, int restart) {
        double prob;
        double bestLogProb = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < restart; i++) {
            HMM.firstRestart = true;
            HMM.seq = seq;
            HMM.train(iter, rng.nextInt(), false, false);
            prob = HMM.logProb;
            if (prob > bestLogProb) {
                bestLogProb = prob;
            }
        }
        return bestLogProb;
    }

    public static void makeA(int dim, int textLen, int iter, int restart, int seed, String storeDir) throws Exception {
        BufferedWriter out = new BufferedWriter(new FileWriter(storeDir));
        Random rng = new Random(seed);
        int[] text = Helper.textProcess("brown_nolines.txt", true, 100000, textLen);
        TriHMM HMM = new TriHMM(null, dim, 26, text);
        double prob;
        double bestLogProb = Double.NEGATIVE_INFINITY;
        double[][][] bestA = new double[dim][dim][dim];
        double[][] bestB = new double[dim][26];
        for (int i = 0; i < restart; i++) {
            HMM.firstRestart = true;
            HMM.train(iter, true, rng.nextInt(), false);
            prob = HMM.logProb;
            System.out.printf("Iteration %d, logProb = %f\n", i+1, prob);
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        out.write(String.format("%f ", HMM.A[j][k][l]));
                    }
                }
                out.newLine();
            }
            out.newLine();
            if (bestLogProb < prob) {
                bestLogProb = prob;
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k < dim; k++) {
                        for (int l = 0; l < dim; l++) {
                            bestA[j][k][l] = HMM.A[j][k][l];
                        }
                    }
                }
                bestB = Helper.deepClone(HMM.B);
            }
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    out.write(String.format("%f ", bestA[i][j][k]));
                }
            }
            out.newLine();
        }
        out.newLine();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 26; j++) {
                out.write(String.format("%f ", bestB[i][j]));
            }
            out.newLine();
        }
        out.close();
    }

    public static int[][] readCipherMatrix(String dir) throws Exception{
        Scanner in = new Scanner(new BufferedReader(new FileReader(dir)));
        int[][] mat = new int[17][24];
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 17; j++) {
                mat[j][i] = in.nextInt();
            }
        }
        return mat;
    }

    public static int[] nextPair(int[] prevPair) {
        int[] nextPair = new int[2];
        if (prevPair[1] == 16) {
            if (prevPair[0] == 0) {
                return null; // all permutations done
            }
            nextPair[0] = 0;
            nextPair[1] = prevPair[1] - prevPair[0] + 1;
            return nextPair;
        }
        nextPair[0] = prevPair[0] + 1;
        nextPair[1] = prevPair[1] + 1;
        return nextPair;
    }

    public static int[] swapCol(int[] swapPair, int[] col) {
        int[] ret = new int[col.length];
        for (int i = 0; i < col.length; i++) {
            ret[i] = col[i];
        }
        int t = ret[swapPair[1]];
        ret[swapPair[1]] = ret[swapPair[0]];
        ret[swapPair[0]] = t;
        return ret;
    }

    public static int[] flattenMatrix(int[][] mat, int[] col) {
        assert (col.length == 17);
        int[] ret = new int[408];
        int idx = 0;
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 17; j++) {
                ret[idx] = mat[col[j]][i];
                idx += 1;
            }
        }
        return ret;
    }
}
