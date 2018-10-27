package em;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import util.Helper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Scanner;

public class EMExpectationTest {
    public static void main(String[] args) throws Exception {

        int nTag = 26;
        int nWord = 26;
        int T = 408;

        String cipherDir = "simple_cipher/simple_408";
        Scanner reader = new Scanner(new BufferedReader(new FileReader(cipherDir)));
        int[] pt = new int[T];
        for (int i = 0; i < T; i++) {
            pt[i] = reader.nextInt();
        }
        int[] ct = new int[T];
        for (int i = 0; i < T; i++) {
            ct[i] = reader.nextInt();
        }
        double[][][] trigram = Helper.readTrigram("data/trigram");
        double[][][][] quadgram = Helper.readQuadgram("data/quadgram");
//        for (int i = 0; i < nTag; i++) {
//            for (int j = 0; j < nTag; j++) {
//                for (int k = 0; k < nTag; k++) {
//                    double s=0;
//                    for (int l = 0; l < nTag; l++) {
//                        s += quadgram[i][j][k][l];
//                    }
//                    if(Math.abs(s - 1) > 0.0000001) {
//                        System.out.println("A does not sum to 1");
//                        System.out.println(s);
//                    };
//                    for (int l = 0; l < nTag; l++) {
//                        if(quadgram[i][j][k][l] < 0.0000001) {
//                            System.out.println("0 in A");
//                            System.out.println(quadgram[i][j][k][l]);
//                        };
//                    }
//                }
//            }
//        }
        long start = System.nanoTime();

//        RandomGenerator rng = new MersenneTwister(-3076155353333121539L);
        RandomGenerator rng = new MersenneTwister(999L);
        Settings settings = new Settings(200, 0,true);
//        ExpectationEstimator estimator = new GibbsEstimator(trigram, nTag, nWord, ct, rng, settings);
//        ExpectationEstimator estimator = new ForwardBackwardEstimator(trigram, nTag, nWord, ct, rng);
        ExpectationEstimator estimator = new QuadGibbsEstimator(quadgram, nTag, nWord, ct, rng, settings);
        EM em = new EM(nTag, nWord, ct, estimator, rng, settings);
        em.train(50);

        long stop = System.nanoTime();
//        FHMM.train(200, false, 8781939572407739913L, true);
//        FHMM.train(200, false, 1209845257843231593L, true);
//        FHMM.train(200, false, 3738420990656387694L, true);

        stop = System.nanoTime();
        System.out.printf("training time: %f\n", (stop - start) / 1e9);

        /*
        Use highest-by-column decode
         */
        int[] argmax = new int[nWord];
        for (int k = 0; k < nWord; k++) {
            double max = em.getB()[0][k];
            int arg = 0;
            for (int l = 1; l < nTag; l++) {
                if (max < em.getB()[l][k]) {
                    max = em.getB()[l][k];
                    arg = l;
                }
            }
            argmax[k] = arg;
        }
        int[] sol_num = new int[T];
        double acc = 0;
        for (int k = 0; k < T; k++) {
            sol_num[k] = argmax[ct[k]];
            if (sol_num[k] == pt[k]) {
                acc++;
            }
        }
        int[] decode = new int[T];
        for (int i = 0; i < T; i++) {
            decode[i] = argmax[ct[i]];
        }
        System.out.println(Helper.display2(em.getB()));
        System.out.println(Arrays.toString(Helper.toCharacterSequence(decode)));
        acc /= T;
        System.out.println(acc);
    }
}
