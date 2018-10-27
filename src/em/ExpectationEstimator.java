package em;

import math.CategoricalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import java.util.*;

public abstract class ExpectationEstimator {

    double[][] B;
    int len;
    int nt;
    int nw;
    double[] logProb;
    int[] obs;
    RandomGenerator rng;
    Settings settings;

    CategoricalDistribution cd;
    public ExpectationEstimator(int nTag, int nWord, int[] obs, RandomGenerator rng, Settings settings) {
        this.obs = obs;
        len = obs.length;

        nt = nTag;
        nw = nWord;
        B = new double[nt][nw];
        this.rng = rng;
        this.settings = settings;
        cd = new CategoricalDistribution(rng);
    }

    public abstract double[][] estimate();

    public double logProb(int[] sample, double[][][] A){
        double lp = 0;
        for (int i = 2; i < len; i++) {
            lp += FastMath.log(A[sample[i-2]][sample[i-1]][sample[i]]);
        }
        return lp;
    }

    public void setB(double[][] B){
        this.B = B;
    }

    /*
    Randomly create a first key
     */
    public int[] firstSample() {
        String mostCommon = "etaoinshrdlcumwfgypbvkjxqz";
        HashMap<Integer, LinkedList<Integer>> dict = new HashMap<>();
        for (int i = 0; i < len; i++) {
            if (!dict.containsKey(obs[i])) {
                dict.put(obs[i], new LinkedList<Integer>());
            }
            dict.get(obs[i]).add(i);
        }
        List<Map.Entry<Integer, LinkedList<Integer>>> list = new ArrayList(dict.entrySet());
        Collections.sort(list, (a, b) -> b.getValue().size() - a.getValue().size()); // descending

        int[] sample = new int[len];
        int rank = 0;
        for (Map.Entry<Integer, LinkedList<Integer>> entry : list) {
            int o = entry.getKey();
            int x = mostCommon.charAt(rank % mostCommon.length()) - 97;
            assert(x < 26);
            for (int idx : entry.getValue()) {
                sample[idx] = x;
            }
            rank += 1;
        }
        return sample;
    }
}
