package mcmc;


import util.Helper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.DoubleStream;

/**
 * Compare the expectation of hidden state computed from forward-backward with that averaged from random samples
 */

public class ExpectationComparison {
    public static void main(String[] args) throws Exception {
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
        double[][][] trigram = Helper.readTrigram("data/trigram");
        HMMBlockSampler sampler = new HMMBlockSampler(trigram, nTag, nWord, ct);
        int[] firstSample = sampler.firstSample();
        sampler.emissionSampling(firstSample);
        sampler.sanityCheck();
        sampler.forward();


        int nIter = 1000;
        int burnin = 100;

        double[][] trueExpection = sampler.forwardBackward();
        double[][] approxBlockExpectation = sampler.blockSamplingProcedure(nIter, burnin, sampler.firstSample(), false);
        double[][] approxGibbsExpectation = sampler.gibbsSamplingProcedure(nIter, burnin, sampler.firstSample(), false);


        System.out.println(Arrays.toString(trueExpection[5]));
        System.out.println(DoubleStream.of(trueExpection[5]).summaryStatistics());
        System.out.println(Arrays.toString(approxBlockExpectation[5]));
        System.out.println(DoubleStream.of(approxBlockExpectation[5]).summaryStatistics());
        System.out.println(Arrays.toString(approxGibbsExpectation[5]));
        System.out.println(DoubleStream.of(approxGibbsExpectation[5]).summaryStatistics());

    }
}
