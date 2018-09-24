import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

public class makeA {
    public static void makeTrigramA(int dim, int textLen, int iter, int restart, int seed, String storeDir) throws Exception {
        BufferedWriter out = new BufferedWriter(new FileWriter(storeDir));
        Random rng = new Random(seed);
        int[] text = util.textProcess("brown_nolines.txt", true, 100000, textLen);
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
                bestB = util.deepClone(HMM.B);
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

    public static void makeBigramA(int dim, int textLen, int iter, int restart, int seed, String storeDir) throws Exception {
        BufferedWriter out = new BufferedWriter(new FileWriter(storeDir));
        Random rng = new Random(seed);
        int[] text = util.textProcess("brown_nolines.txt", true, 100000, textLen);
        double[] alpha = new double[26];
        for (int i = 0; i < 26; i++) {
            alpha[i] = 1.01;
        }
        MDir prior = new MDir(26, alpha, 0.0001);
        BayesianHMM HMM = new BayesianHMM(dim, 26, text, prior);
        double prob;
        double bestLogProb = Double.NEGATIVE_INFINITY;
        double[][] bestA = new double[dim][dim];
        double[][] bestB = new double[dim][26];
        for (int i = 0; i < restart; i++) {
            HMM.firstRestart = true;
            HMM.train(iter, rng.nextInt(), true, false);
            prob = HMM.logProb;
            System.out.printf("Iteration %d, logProb = %f\n", i+1, prob);
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    out.write(String.format("%f ", HMM.A[j][k]));

                }
                out.newLine();
            }
            out.newLine();
            if (bestLogProb < prob) {
                bestLogProb = prob;
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k < dim; k++) {
                        bestA[j][k] = HMM.A[j][k];
                    }
                }
                bestB = util.deepClone(HMM.B);
            }
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                out.write(String.format("%f ", bestA[i][j]));
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

    public static void main(String[] args) throws Exception {
        makeBigramA(4, 10000, 200, 100, 12345, "biA4_restarts");
    }
}
