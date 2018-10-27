import em.HMM;
import em.map.MapHMM;
import em.map.MDir;
import em.map.MDir_orig;
import util.Helper;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by drproduck on 11/18/17.
 */
public class main {
    public static void main(String[] args) throws Exception {
//        sparseDirTest();
//        basicExample();
//        testEnglish();

//        String dir = "/home/drproduck/Documents/em.HMM/simple_cipher/simple_cipher_length=300";
//        System.out.println("Solving cipher of length 300");
//        simpleCipher(300, dir);
//        dir = "/home/drproduck/Documents/em.HMM/simple_cipher/simple_cipher_length=400";
//        System.out.println("Solving cipher of length 400");
//        simpleCipher(400, dir);
//        dir = "/home/drproduck/Documents/em.HMM/simple_cipher/simple_cipher_length=500";
//        System.out.println("Solving cipher of length 500");
//        simpleCipher(500, dir);

        // use these 2 lines only
//        cipher408();
        cipher408Bayesian();


//        BufferedWriter writer = new BufferedWriter(new FileWriter("homophonic_408"));
//        int[] plain = util.Helper.plain("plain.txt");
//        for (int i = 0; i < 408; i++) {
//            writer.write(plain[i]+" ");
//        }
//        writer.newLine();
//        int[] cipher = util.Helper.read408("408t");
//        for (int i = 0; i < 408; i++) {
//            writer.write(cipher[i] + " ");
//        }
//        writer.close();
    }

    /**
     * This one simply adds the smoothing constant
     * @throws Exception
     */
    public static void cipher408() throws Exception {
        Random rng = new Random(9999);
        Scanner sc = new Scanner(new BufferedReader(new FileReader("/home/drproduck/Documents/em.HMM/data/bigram")));

        double[][] A = new double[26][26];
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                A[i][j] = sc.nextDouble();
            }
        }
        int n_tags = 26;
        int n_words = 54;
        int maxIters = 200;
        int T = 408;
        int[] cipher = Helper.read408("/home/drproduck/Documents/em.HMM/data/408ciphercleaned");
        int[] plain = Helper.plain("/home/drproduck/Documents/em.HMM/data/408plaincleaned");

        double[] alpha = new double[n_words];
        for (int i = 0; i < n_words; i++) {
            alpha[i] = 1.01;
        }

        HMM hmm = new HMM(n_tags, n_words, cipher);
        hmm.A = A;
        int[] n = {1000};

        double curProb = 0;
        double[][] saveB = new double[n_tags][n_words];
        for (int i = 0; i < n.length; i++) {
            System.out.println("n = " + n[i]);
            System.out.println("Restart LogProb KeyAccuracy Seed Time");
            double best = Double.NEGATIVE_INFINITY;
            double bestKey = -1.0;
            double bestModelKey = -1.0;
            double keyScore;
            String bestModelSolution = null;
            for (int j = 0; j < n[i]; j++) {
                long seed = rng.nextLong();
                hmm.firstRestart = true;
                long start = System.nanoTime();
                hmm.train(maxIters, seed, false, false);
                long stop = System.nanoTime();
                curProb = hmm.logProb;

//                int[] argmax = new int[n_words];
//                for (int k = 0; k < n_words; k++) {
//                    double max = hmm.B[0][k];
//                    int arg = 0;
//                    for (int l = 1; l < n_tags; l++) {
//                        if (max < hmm.B[l][k]) {
//                            max = hmm.B[l][k];
//                            arg = l;
//                        }
//                    }
//                    argmax[k] = arg;
//                }
//                int[] sol_num = new int[T];
//                double acc = 0;
//                for (int k = 0; k < T; k++) {
//                    sol_num[k] = argmax[cipher[k]];
//                    if (sol_num[k] == plain[k]) {
//                        acc++;
//                    }
//                }
//                acc /= T;
                double acc = 0;
                int[] sol_num = hmm.viterbi();
                for (int k = 0; k < T; k++) {
                    if (sol_num[k] == plain[k]) {
                        acc += 1;
                    }
                }
                acc /= T;

                System.out.printf("%d %f %f %d %f\n", j, curProb, acc, seed, (stop - start) * 1.0 / 1e9);
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < T; k++) {
                    sb.append(Character.toString((char) (sol_num[k] + 65)));
                }
                String solution = sb.toString();
                System.out.println(solution);
                if (best < curProb) {
                    best = curProb;
                    bestModelKey = acc;
                    bestModelSolution = solution;
                    saveB = Helper.deepClone(hmm.B);
                }
                if (bestKey < acc) {
                    bestKey = acc;
                }
            }
            System.out.printf("Best model's score = %f, Best model's key ac = %f, Best possible key ac = %f\n", best, bestModelKey, bestKey);
            System.out.println(bestModelSolution);
            System.out.println(Helper.display2(saveB));
        }
    }

    /**
     * This one uses em.map.MapHMM and em.map.MDir
     * @throws Exception
     */
    public static void cipher408Bayesian() throws Exception {
        Random rng = new Random(9999);
//        double[][] A = util.Helper.digraph();
        Scanner sc = new Scanner(new BufferedReader(new FileReader("/home/drproduck/Documents/em.HMM/data/bigram")));

        double[][] A = new double[26][26];
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                A[i][j] = sc.nextDouble();
            }
        }
        int n_tags = 26;
        int n_words = 54;
        int maxIters = 200;
        int T = 408;
        int[] cipher = Helper.read408("/home/drproduck/Documents/em.HMM/data/408ciphercleaned");
        int[] plain = Helper.plain("/home/drproduck/Documents/em.HMM/data/408plaincleaned");

        double[] alpha = new double[n_words];
        for (int i = 0; i < n_words; i++) {
            alpha[i] = 1.01;
        }
        MDir prior = new MDir(n_words, alpha, 0.0001);
        MapHMM HMM = new MapHMM(n_tags, n_words, T, prior);
        HMM.A = A;
        HMM.seq = cipher;
        int[] n = {100};

        double curProb = 0;
        double[][] saveB = new double[n_tags][n_words];
        for (int i = 0; i < n.length; i++) {
            System.out.println("n = " + n[i]);
            System.out.println("Restart LogProb KeyAccuracy Seed Time");
            double best = Double.NEGATIVE_INFINITY;
            double bestKey = -1.0;
            double bestModelKey = -1.0;
            double keyScore;
            String bestModelSolution = null;
            for (int j = 0; j < n[i]; j++) {
                int seed = rng.nextInt();
                long start = System.nanoTime();
                HMM.train(maxIters, seed, false, false);
                curProb = HMM.logProb;
                long stop = System.nanoTime();

//                int[] argmax = new int[n_words];
//                for (int k = 0; k < n_words; k++) {
//                    double max = hmm.B[0][k];
//                    int arg = 0;
//                    for (int l = 1; l < n_tags; l++) {
//                        if (max < hmm.B[l][k]) {
//                            max = hmm.B[l][k];
//                            arg = l;
//                        }
//                    }
//                    argmax[k] = arg;
//                }
//                int[] sol_num = new int[T];
//                double acc = 0;
//                for (int k = 0; k < T; k++) {
//                    sol_num[k] = argmax[cipher[k]];
//                    if (sol_num[k] == plain[k]) {
//                        acc++;
//                    }
//                }
//                acc /= T;
                double acc = 0;
                int[] sol_num = HMM.viterbi();
                for (int k = 0; k < T; k++) {
                    if (sol_num[k] == plain[k]) {
                        acc += 1;
                    }
                }
                acc /= T;

                System.out.printf("%d %f %f %d, %f\n", j, curProb, acc, seed, (stop - start) * 1.0 / 1e9);
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < T; k++) {
                    sb.append(Character.toString((char) (sol_num[k] + 65)));
                }
                String solution = sb.toString();
                System.out.println(solution);
                if (best < curProb) {
                    best = curProb;
                    bestModelKey = acc;
                    bestModelSolution = solution;
                    saveB = Helper.deepClone(HMM.B);
                }
                if (bestKey < acc) {
                    bestKey = acc;
                }
            }
            System.out.printf("Best model's score = %f, Best model's key ac = %f, Best possible key ac = %f\n", best, bestModelKey, bestKey);
            System.out.println(bestModelSolution);
            System.out.println(Helper.display2(saveB));
        }
    }

    public static void simpleCipher(int T, String dir) throws IOException {
        Random rng = new Random(1235);
        Scanner sc = new Scanner(new BufferedReader(new FileReader("/home/drproduck/Documents/em.HMM/data/bigram")));
//        double[][] A = util.Helper.digraph();
        double[][] A = new double[26][26];
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                A[i][j] = sc.nextDouble();
            }
        }
        int n_states = 26;
        int n_obs = 26;
        int maxIters = 100;
        Scanner reader = new Scanner(new BufferedReader(new FileReader(dir)));
        int[] plain = new int[T];
        for (int i = 0; i < T; i++) {
            plain[i] = reader.nextInt();
        }
        int[] cipher = new int[T];
        for (int i = 0; i < T; i++) {
            cipher[i] = reader.nextInt();
        }
        double[] alpha = new double[26];
        for (int i = 0; i < 26; i++) {
            alpha[i] = 1;
        }
        MDir prior = new MDir(26, alpha, 0);
        MapHMM HMM = new MapHMM(n_states, n_obs, T, prior);
        HMM.A = A;
        HMM.seq = cipher;
        int[] n = {100};

        double curProb = 0;
        double[][] saveB = new double[26][26];
        for (int i = 0; i < n.length; i++) {
            System.out.println("n = " + n[i]);
            System.out.println("Restart Score KeyAccuracy");
            double best = Double.NEGATIVE_INFINITY;
            double bestKey = -1.0;
            double bestModelKey = -1.0;
            double keyScore;
            String bestModelSolution = null;
            for (int j = 0; j < n[i]; j++) {
                HMM.train(maxIters, rng.nextInt(), false, false);
                curProb = HMM.logProb;

                int[] argmax = new int[26];
                for (int k = 0; k < 26; k++) {
                    double max = HMM.B[0][k];
                    int arg = 0;
                    for (int l = 1; l < 26; l++) {
                        if (max < HMM.B[l][k]) {
                            max = HMM.B[l][k];
                            arg = l;
                        }
                    }
                    argmax[k] = arg;
                }
                int[] sol_num = new int[T];
                double acc = 0;
                for (int k = 0; k < T; k++) {
                    sol_num[k] = argmax[cipher[k]];
                    if (sol_num[k] == plain[k]) {
                        acc ++;
                    }
                }
                acc /= T;
                System.out.printf("%d %f %f\n", j, curProb, acc);
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < T; k++) {
                    sb.append(Character.toString((char) (sol_num[k] + 65)));
                }
                String solution = sb.toString();
                if (best < curProb) {
                    best = curProb;
                    bestModelKey = acc;
                    bestModelSolution = solution;
                    saveB = Helper.deepClone(HMM.B);
                }
                if (bestKey < acc) {
                    bestKey = acc;
                }
            }
            System.out.printf("Best model's score = %f, Best model's key ac = %f, Best possible key ac = %f\n", best, bestModelKey, bestKey);
            System.out.println(bestModelSolution);
            System.out.println(Helper.display2(saveB));
        }
    }

//    public static int postMap(int[] sol_num, int[] plain, int[] argmax, int T) {
//        double acc = 0;
//        for (int k = 0; k < T; k++) {
//            sol_num[k] = argmax[cipher[k]];
//            if (sol_num[k] == plain[k]) {
//                acc ++;
//            }
//        }
//        acc /= T;
//        return acc;
//    }

    public static void testEnglish() throws Exception {
        int[] txt_obs = Helper.textProcess("/home/drproduck/Documents/hmm/src/brown_nolines.txt", true, 100000, 100000);
        System.out.println("text processed successfully");
        double[] alpha = new double[26];
        for (int i = 0; i < 26; i++) {
            alpha[i] = -5;
        }
        MDir prior = new MDir(26, alpha, 0.001);
        MapHMM trainer = new MapHMM(26, 26, txt_obs, prior);
        System.out.println("declare successfully");
        trainer.train(100, 5678, true, true);
        BufferedWriter outputWriter = new BufferedWriter(new FileWriter("output2sparse.txt"));
        outputWriter.write(Arrays.deepToString(trainer.A));
        outputWriter.newLine();
        outputWriter.write(Helper.display2(trainer.B));
        outputWriter.newLine();
        outputWriter.write(Arrays.toString(trainer.pi));
        outputWriter.newLine();
        outputWriter.flush(); outputWriter.close();
    }

    public static void basicExample() {
        MDir dirPrior = new MDir(3, new double[]{-4, -4, -4}, 0.01);
        MapHMM test = new MapHMM(2, 3, 3, dirPrior);
        test.seq = new int[]{1, 0, 2};
        test.A = new double[][]{{0.7, 0.3}, {0.4, 0.6}};
//        test.B = new double[][]{{0.1, 0.4, 0.5}, {0.7, 0.2, 0.1}};
        test.pi = new double[]{0.5, 0.5};
        test.train(100, 9999, false, true);
        System.out.println(Arrays.deepToString(test.B));

//        System.out.println(Arrays.toString(test.viterbi()));
//        System.out.println(Arrays.toString(test.besthmm()));
//        test.prob3(3, 3);
//        System.out.println(test.sc);
    }

    public static void sparseDirTest() {
        MDir dir = new MDir(3, new double[]{-4,-4,-4}, 0.01);
        MDir_orig dir1 = new MDir_orig(3, new double[]{-4, -3, -2}, 0.01);

        System.out.println(Arrays.toString(dir.getMode(new double[]{7, 8, 9})));
        System.out.println(Arrays.toString(dir1.getMode()));

    }
}
