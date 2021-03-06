package em;

import java.io.*;
import java.util.Random;
import java.util.Scanner;
import util.Helper;

public class trigramtest {
    public static void main(String[] args) throws Exception{
        String cipherDir = "simple_cipher/simple_cipher_length=300";
        String logtrigramDir = "data/logtrigram.txt";
        String trigramDir = "data/em.txt";
//        simpleCipher(300, trigramDir, cipherDir);
        cipher408(trigramDir);
    }

    public static void cipher408(String trigramDir) throws Exception {
        String cipherDir = "data/408ciphercleaned";
        String plainDir = "data/408plaincleaned";
        Random rng = new Random(2018);
//        double[][] A = Helper.Helper.digraph();
        double[][][] A = Helper.readTrigram(trigramDir);
        int nt = 26;
        int nw = 54;
        int iter = 100;
//
        int len = 408;
        int[] cipher = Helper.read408(cipherDir);
        int[] plain = Helper.plain(plainDir);
//


//        int T = 390;
//        Scanner in = new Scanner(new BufferedReader(new FileReader(cipherDir)));
//        int[] cipher = new int[T];
//        for (int i = 0; i < T; i++) {
//            cipher[i] = in.nextInt() - 1;
//        }
//        System.out.println(Arrays.toString(cipher));
//        in = new Scanner(new BufferedReader(new FileReader(plainDir)));
//        int[] plain = new int[T];
//        for (int i = 0; i < T; i++) {
//            plain[i] = in.next().charAt(0) - 65;
//        }

        FastTHMM HMM = new FastTHMM(A, nt, nw, cipher);
        int[] n = {500};

        double curProb = 0;
        double[][] saveB = new double[nt][nw];
        for (int i = 0; i < n.length; i++) {
            System.out.println("n = " + n[i]);
            System.out.println("Restart Seed Score KeyAccuracy Time");
            double best = Double.NEGATIVE_INFINITY;
            double bestKey = -1.0;
            double bestModelKey = -1.0;
            double keyScore;
            String bestModelSolution = null;
            for (int j = 0; j < n[i]; j++) {
                HMM.firstRestart = true;
                Long seed = rng.nextLong();
                long start = System.nanoTime();
                HMM.train(iter, false, seed, false);
                long stop = System.nanoTime();
                curProb = HMM.logProb;
//                int[] argmax = new int[nw];
//                for (int k = 0; k < nw; k++) {
//                    double max = em.HMM.B[0][k];
//                    int arg = 0;
//                    for (int l = 1; l < nt; l++) {
//                        if (max < em.HMM.B[l][k]) {
//                            max = em.HMM.B[l][k];
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

                int[] sol = HMM.viterbi();
                double ac = 0;
                for (int k = 0; k < len; k++) {
                    if (sol[k] == plain[k]) {
                        ac += 1;
                    }
                }
                ac /= len;

                System.out.printf("%d %d %f %f %f\n", j, seed, curProb, ac, (stop - start) * 1.0 / 1e9);
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < len; k++) {
                    sb.append(Character.toString((char) (sol[k] + 65)));
                }
                String solution = sb.toString();
                System.out.println(solution);
                if (best < curProb) {
                    best = curProb;
                    bestModelKey = ac;
                    bestModelSolution = solution;
                    saveB = Helper.deepClone(HMM.B);
                }
                if (bestKey < ac) {
                    bestKey = ac;
                }
            }
            System.out.printf("Best model's score = %f, Best model's key ac = %f, Best possible key ac = %f\n", best, bestModelKey, bestKey);
            System.out.println(bestModelSolution);
            System.out.println(Helper.display2(saveB));
        }
    }

    public static void simpleCipher(int T, String trigramDir, String cipherDir) throws Exception {
        Random rng = new Random(2018);
        double[][][] A = Helper.readTrigram(trigramDir);
        int nTag = 26;
        int nWord = 26;
        int maxIters = 100;
        Scanner reader = new Scanner(new BufferedReader(new FileReader(cipherDir)));
        int[] plain = new int[T];
        for (int i = 0; i < T; i++) {
            plain[i] = reader.nextInt();
        }
        int[] cipher = new int[T];
        for (int i = 0; i < T; i++) {
            cipher[i] = reader.nextInt();
        }

        FastTHMM hmm = new FastTHMM(A, nTag, nWord, cipher);
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
                hmm.firstRestart = true;
                long seed = rng.nextLong();
                long start = System.nanoTime();
                hmm.train(maxIters, false, seed, false);
                long stop = System.nanoTime();
                curProb = hmm.logProb;
                int[] argmax = new int[26];
                for (int k = 0; k < 26; k++) {
                    double max = hmm.B[0][k];
                    int arg = 0;
                    for (int l = 1; l < 26; l++) {
                        if (max < hmm.B[l][k]) {
                            max = hmm.B[l][k];
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
}
