package mcmc;

import java.io.BufferedWriter;
import java.io.FileWriter;
import util.Helper;
import java.util.Random;

public class GibbsMain {
    public static void main(String[] args) throws Exception {
//        solveSimple();
        solve408(100);
    }

    public static void solveSimple() throws Exception {
        double[] pi = Helper.readPi("/home/drproduck/Documents/em.HMM/data/monogram");
        double[][] bi = Helper.readBigram("/home/drproduck/Documents/em.HMM/data/bigram");
        int length = 1000;
        int[] obs = Helper.textProcess("/home/drproduck/Documents/hmm/src/brown_nolines.txt", true, 1000, length);
        int[] dict = Helper.permuteKey("caesar", 1234);
        int[] cipher = new int[obs.length];
        for (int i = 0; i < obs.length; i++) {
            cipher[i] = dict[obs[i]];
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            sb.append(Character.toString((char) (obs[i] + 97)));
        }
        System.out.println(sb.toString());
        Random r = new Random(9999);
        int[] tags = new int[length];
        for (int i = 0; i < length; i++) {
            tags[i] = (int) (r.nextDouble() * 26);
        }
        BayesGibbs sampler = new BayesGibbs(pi, bi, tags, obs, 26, 26);
        sampler.typeSampler(5000, "output.txt");
    }

    public static void solve408(int n_restart) throws Exception {
        double[] pi = Helper.readPi("/home/drproduck/Documents/em.HMM/data/monogram");
        double[][] bi = Helper.readBigram("/home/drproduck/Documents/em.HMM/data/bigram");
        Random r = new Random(9999);
        int length = 408;
        int[] words = Helper.read408("cipher.txt");
        int[] plain = Helper.plain("plain.txt");
        BufferedWriter writer = new BufferedWriter(new FileWriter("/home/drproduck/Documents/em.HMM/pointSampler.txt"));
        for (int t = 0; t < n_restart; t++) {
            System.out.println("restart "+t);
            int[] tags = new int[length];
            for (int i = 0; i < length; i++) {
                tags[i] = (int) (r.nextDouble() * 26);
            }

            BayesGibbs sampler = new BayesGibbs(pi, bi, tags, words, 26, 53);
            sampler.pointSampler(5000, String.format("pointSamper_408_%d.txt", t));
            int[] sol = sampler.tags;
            double acc = 0;
            for (int i = 0; i < length; i++) {
                if (sol[i] == plain[i]) {
                    acc ++;
                }
            }
            System.out.println(acc / 408);
            writer.write("" + acc + " " + sampler.getLogProb()+"\n");
        }
        writer.close();
    }
}
