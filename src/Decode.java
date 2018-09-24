import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;

public class Decode {

    public static void main(String[] args) throws Exception{
        double[][] transition = util.readBigram("data/bigram");
        double[] pi = util.readPi("data/monogram");
        int[] observations = util.read408("408.txt");
        int[] plain = util.plain("plain.txt");
        int length = 408;
        int n_tags = 26;
        double[][] emission = new double[26][53];

        Scanner scanner;
        File dir = new File("/home/drproduck/Documents/HMM/bfgs_proposals/proposals");
        File[] directoryListing = dir.listFiles();
        String arg = null;
        double max = -1;
        for (File child : directoryListing) {
            scanner = new Scanner(new BufferedReader(new FileReader(child)));
            scanner.next();
            double logprob = scanner.nextDouble();
            scanner.next();
            String sol = scanner.next();
            double acc = 0;

            for (int i = 0; i < 408; i++) {
                if (sol.charAt(i) - 65 == plain[i]) {
                    acc ++;
                }
            }
            acc = acc / 408;
            if (acc > max) {
                max = acc;
                arg = child.getName();
            }
        }
        System.out.println(max);
    }

    public static int[] viterbi(double[] pi, double[][] transition, double[][] emission, int[] observations, int length, int n_tags) {
        double c = 0;

        int[][] dp = new int[length][n_tags];
        double[] head = new double[n_tags];

        for (int i = 0; i < n_tags; i++) {
            head[i] = Math.log(pi[i]) + 3 * Math.log(emission[i][observations[0]]);
        }

        for (int t = 1; t < length; t++) {
            for (int i = 0; i < n_tags; i++) {
                double x = 0;
                double max = Double.NEGATIVE_INFINITY;
                int argmax = -1;
                for (int j = 0; j < n_tags; j++) {
                    x = head[j] + Math.log(transition[j][i]);
                    if (x > max) {
                        max = x;
                        argmax = j;
                    }
                }
                dp[t][i] = argmax;
                head[i] = max + 3 * Math.log(emission[i][observations[t]]);
            }
        }

        //get the last state of the best DP sequence
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n_tags; i++) {
            if (head[i] > max) {
                max = head[i];
                argmax = i;
            }
        }
        //trace back
        int[] res = new int[length];
        res[length -1] = argmax;
        for (int t = length -1; t > 0; t--) {
            argmax = dp[t][argmax];
            res[t-1] = argmax;
        }

        return res;
    }
}
