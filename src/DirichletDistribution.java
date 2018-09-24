import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.Map;

public class DirichletDistribution {
    RandomGenerator rng = new MersenneTwister();
    double[] prior;
    public DirichletDistribution(RandomGenerator rng, double[] prior){
        this(prior);
        this.rng = rng;
    }
    public DirichletDistribution(double[] alpha){
        this.prior = alpha;
    }

    public double[] sample(double[] empiricalCounts) {
        double[] res = new double[prior.length];
        double sum = 0;
        for (int i = 0; i < prior.length; i++) {
            res[i] = new GammaDistribution(rng, prior[i] + empiricalCounts[i], 1).sample();
            sum += res[i];
        }
        for (int i = 0; i < prior.length; i++) {
            res[i] /= sum;
        }
        return res;
    }
    public double[] sample(Counters<Integer> empiricalCounts){

        double[] res = new double[prior.length];
        double sum = 0;
        for (int i = 0; i < prior.length; i++) {
            res[i] = new GammaDistribution(rng, prior[i] + empiricalCounts.getCount(i), 1).sample();
            sum += res[i];
        }
        for (int i = 0; i < prior.length; i++) {
            res[i] /= sum;
        }
        return res;
    }
}
