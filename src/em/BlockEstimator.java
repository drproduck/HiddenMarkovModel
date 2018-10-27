package em;


import org.apache.commons.math3.random.RandomGenerator;
import util.Pair;


import static util.MatrixHelper.firstSlice;
import static util.MatrixHelper.flattenMatrixSampleIndex;

public class BlockEstimator extends ExpectationEstimator{
    double[][][] A;
    double[][][] alpha;
    double[] c;
    double[][][][] trigamma;
    public BlockEstimator(double[][][] transition, int nTag, int nWord, int[] obs, RandomGenerator rng, Settings settings) {
        super(nTag, nWord, obs, rng, settings);
        A = transition;
        alpha = new double[len - 1][nt][nt];
        c = new double[len - 1];
        trigamma = new double[len][nt][nt][nt];
    }

    public void forward() {

        this.c = new double[len-1];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                alpha[0][i][j] = B[i][obs[0]] * B[j][obs[1]];
                c[0] += alpha[0][i][j];
            }
        }
        c[0] = 1 / c[0];
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                alpha[0][i][j] *= c[0];
            }
        }

        for (int t = 1; t < len - 1; t++) {
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    for (int k = 0; k < nt; k++) {
                        trigamma[t][i][j][k] = alpha[t-1][i][j] * A[i][j][k] * B[k][obs[t+1]];
                        alpha[t][j][k] += trigamma[t][i][j][k];
                        c[t] += alpha[t][j][k];
                    }
                }
            }
            // normalize
            c[t] = 1 / c[t];
            for (int i = 0; i < nt; i++) {
                for (int j = 0; j < nt; j++) {
                    alpha[t][i][j] *= c[t];
                }
            }
        }
    }

    public int[] backwardSampling() {
        int[] sample = new int[len];
        Pair<Integer, Integer> idx = flattenMatrixSampleIndex(alpha[len-2], cd);
        sample[len-2] = idx.a;
        sample[len-1] = idx.b;
        for (int i = len-3; i >= 0; i--) {
            sample[i] = cd.sample(firstSlice(trigamma[i+1], sample[i+1], sample[i+2]));
        }
        return sample;
    }
    @Override
    public double[][] estimate(){
        logProb = new double[settings.nIter];
        long start, stop;
        int[] hidden = firstSample();
        double[][] expectation = new double[len][nt];

        for (int it = 0; it < settings.nIter; it++) {
            if (settings.verbose) System.out.printf("iteration: %d\n", it);
            forward();
            hidden = backwardSampling();
            if (it >= settings.burnin){
                for (int j = 0; j < len; j++) {
                    expectation[j][hidden[j]] += 1;
                }
            }
        }
        for (int j = 0; j < len; j++) {
            for (int i = 0; i < nt; i++) {
                expectation[j][i] /= (settings.nIter - settings.burnin);
            }
        }
        return expectation;
    }
}
