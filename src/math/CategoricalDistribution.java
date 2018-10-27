package math;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.Arrays;

public class CategoricalDistribution {
    static RandomGenerator rng = new MersenneTwister();
    public CategoricalDistribution(RandomGenerator rng) {
        this.rng = rng;
    }

    public CategoricalDistribution() {}

    public int sample(double[] weights){
        double[] cumweights = new double[weights.length];
        cumweights[0] = weights[0];
        for (int i = 1; i < weights.length; i++) {
            cumweights[i] = cumweights[i-1] + weights[i];
        }
        double val = rng.nextDouble() * cumweights[cumweights.length-1];
        return java.lang.Math.abs(Arrays.binarySearch(cumweights, val)) - 1;
    }

    public static void main(String[] args) {
        Counters<Integer> counter = new Counters<>();
        CategoricalDistribution cd = new CategoricalDistribution();
        double[] weights = new double[]{0,0,3,2,0,4,1};
        for (int i = 0; i < 10000000; i++) {
            counter.increment(cd.sample(weights));
        }
        for (int i = 0; i < weights.length; i++) {
            System.out.println(i+" "+counter.getCount(i)* 1.0 / counter.total);
        }
    }
}

