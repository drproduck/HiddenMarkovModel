import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by drproduck on 11/20/17.
 */
public class Gibbs {
    double[] pi;
    double[][] baseTransit;
    int length;
    int[] tags;
    int[] words;
    int n_tags;
    int n_words;

    double alpha = 1, beta = 0.01; // dirichlet parameters
    int[][] bigramCount;
    int[][] emitCount;
    int[] monoCount;
    double temperature = 1;
    double minTemperature = 1e-4;
    double decay = 0.0001;
    RandomGenerator r = new MersenneTwister(1234);
    class Categorical {
        EnumeratedIntegerDistribution sampler;
        double[] prob;
        int[] singletons;
        Categorical(RandomGenerator r, int n, double[] prob) {
            singletons = new int[n];
            for (int i = 0; i < n; i++) {
                singletons[i] = i;
            }
            this.prob = prob;
            sampler = new EnumeratedIntegerDistribution(r, singletons, prob);
        }

        Categorical(RandomGenerator r, List<Integer> base) {
            int l = base.size();
            this.prob = new double[l];
            for (int i = 0; i < l; i++) {
                prob[i] = 1.0 / l;
            }
            this.singletons = new int[l];
            for (int i = 0; i < l; i++) {
                singletons[i] = base.get(i);
            }
            sampler = new EnumeratedIntegerDistribution(r, singletons, prob);
        }
        int sample() {
            return sampler.sample();
        }

        int getMode() {
            double max = prob[0];
            int argmax = 0;
            for (int i = 1; i < prob.length; i++) {
                if (max < prob[i]) {
                    max = prob[i];
                    argmax = i;
                }
            }
            return argmax;
        }
    }
    Categorical wordSampler;
    Categorical tagSampler;

    public Gibbs(double[] pi, double[][] baseTransit, int[] tags, int[] words, int n_tags, int n_words) {
        this.pi = pi;
        this.baseTransit = baseTransit;
        this.tags = tags;
        this.words = words;
        this.n_tags = n_tags;
        this.n_words = n_words;
        assert (tags.length == words.length);
        length = tags.length;
        initCount();
    }

    public void initCount() {
        bigramCount = new int[n_tags][n_tags];
        emitCount = new int[n_tags][n_words];
        monoCount = new int[n_tags];
        for (int i = 0; i < length-1; i++) {
            bigramCount[tags[i]][tags[i+1]] ++;
            emitCount[tags[i]][words[i]] ++;
            monoCount[tags[i]] ++;
        }
        emitCount[tags[length-1]][words[length-1]] ++;
        monoCount[tags[length-1]] ++;
    }

//    public void checkCount() {
//        double[][] saveEmit = util.deepClone(emitCount);
//        dou
//    }

    public void typeSampler(int iter, String dir) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(dir));
        List<Integer>[] wordType = new ArrayList[n_words];
        for (int i = 0; i < length; i++) {
            if (wordType[words[i]] == null) {
                wordType[words[i]] = new ArrayList<>();
            }
            wordType[words[i]].add(i);
        }
        printTags(-1);
        double[] transitProb;
        int[] prevToCurrCount = new int[n_tags], currToNextCount = new int[n_tags], tagEmit = new int[n_tags];
        int prevTag, nextTag, oldTag, newTag, word;
        for (int it = 0; it < iter; it++) {
            temperature -= decay;
            temperature = (temperature > minTemperature) ? temperature : minTemperature;

            for (int i = 0; i < wordType.length; i++) {
                // randomly pick a position to re-sample
                if (wordType[i] == null) break;
                wordSampler = new Categorical(r, wordType[i]);
                int samplingPos = wordSampler.sample();
                transitProb = new double[n_tags];
                if (samplingPos == 0) {
                    prevTag = -1;
                } else prevTag = tags[samplingPos-1];
                oldTag = tags[samplingPos];
                if (samplingPos == length - 1) {
                    nextTag = -1;
                } else nextTag = tags[samplingPos + 1];
                word = words[samplingPos];

                if (samplingPos != 0) {
                    prevToCurrCount = getPrevToCurrCount(prevTag);
                }
                if (samplingPos != length - 1) {
                    currToNextCount = getCurrToNextCount(nextTag);
                }
                tagEmit = getEmitCount(word);

                // for each tag j, produce a conditional probability
                for (int j = 0; j < n_tags; j++) {
                    // word emit
                    transitProb[j] = (tagEmit[j] - beta) / (monoCount[j] - n_words * beta);

                    // prev to curr
                    if (samplingPos != 0) {
                        transitProb[j] *= (prevToCurrCount[j] + alpha * baseTransit[prevTag][j]) / (monoCount[prevTag] + alpha);
                    }
                    // curr to next
                    if (samplingPos != length - 1) {
                        transitProb[j] *= (currToNextCount[j] + alpha * baseTransit[j][nextTag]) / (monoCount[j] + alpha);
                    }
                    double oldprob = transitProb[j];
//                    transitProb[j] = Math.pow(transitProb[j], 1.0 / temperature);

                }
                tagSampler = new Categorical(r, n_tags, transitProb);
                newTag = tagSampler.sample();

                for (int j = 0; j < wordType[i].size(); j++) {
                    int pos = wordType[i].get(j);
                    if (pos == 0) {
                        prevTag = -1;
                    } else prevTag = tags[pos-1];
                    oldTag = tags[pos];
                    if (pos == length - 1) {
                        nextTag = -1;
                    } else nextTag = tags[pos + 1];
                    word = words[pos];
                    changeBigramCount(prevTag, nextTag, oldTag, newTag);
                    changeMonoCount(oldTag, newTag);
                    changeEmitCount(word, oldTag, newTag);
                    tags[pos] = newTag;
                }
            }
            if (it % 10 == 0) {
                System.out.println("iteration "+it);
                System.out.println(printTags(-1));
                System.out.println(getLogProb());
            }
            if (dir != null) {
                writer.write("iteration: "+it);
                writer.newLine();
                writer.write(printTags(-1));
                writer.newLine();
                writer.write(String.format("score %f", getLogProb()));
                writer.newLine();
            }
        }
    }

    public void pointSampler(int iter, String dir) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(dir));
        double[] transitProb;
        int[] prevToCurrCount = new int[n_tags], currToNextCount = new int[n_tags], tagEmit = new int[n_tags];
        int prevTag, nextTag, oldTag, newTag;
        for (int it = 0; it < iter; it++) {
            temperature -= decay;
            temperature = (temperature > minTemperature) ? temperature : minTemperature;

            for (int i = 0; i < length; i++) {
                transitProb = new double[n_tags];
                if (i == 0) {
                    prevTag = -1;
                } else prevTag = tags[i-1];
                oldTag = tags[i];
                if (i == length - 1) {
                    nextTag = -1;
                } else nextTag = tags[i + 1];
                int word = words[i];

                if (i != 0) {
                    prevToCurrCount = getPrevToCurrCount(prevTag);
                }
                if (i != length - 1) {
                    currToNextCount = getCurrToNextCount(nextTag);
                }
                tagEmit = getEmitCount(word);

                // for each tag j, produce a conditional probability
                for (int j = 0; j < n_tags; j++) {
                    // word emit
                    transitProb[j] = (tagEmit[j] + beta) / (monoCount[j] + n_words * beta);
                    // prev to curr
                    if (i != 0) {
                        transitProb[j] *= (prevToCurrCount[j] + alpha * baseTransit[prevTag][j]) / (monoCount[prevTag] + alpha);
                    }
                    // curr to next
                    if (i != length - 1) {
                        transitProb[j] *= (currToNextCount[j] + alpha * baseTransit[j][nextTag]) / (monoCount[j] + alpha);
                    }

                    transitProb[j] = Math.pow(transitProb[j], 1.0 / temperature);
                }
                tagSampler = new Categorical(r, n_tags, transitProb);
                newTag = tagSampler.sample();
                tags[i] = newTag;
//                System.out.printf("position %d, old tag = %d, mode = %d, sampled tag = %d\n", i, oldTag, tagSampler.getMode(), newTag);
                changeBigramCount(prevTag, nextTag, oldTag, newTag);
                changeMonoCount(oldTag, newTag);
                changeEmitCount(word, oldTag, newTag);
            }
            if (it % 10 == 0) {
//                System.out.println("iteration "+it);
//                System.out.println(printTags(-1));
//                System.out.println(logProbFromAlpha());
                writer.write(""+getLogProb());
                writer.newLine();
            }
        }
    }

    public String printTags(int pos) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            if (i == pos) {
                sb.append(Character.toString((char) (tags[i]+65)));
            }
            else sb.append(Character.toString((char) (tags[i]+97)));
        }
        return sb.toString();
    }

    public int[] getEmitCount(int word) {
        int[] ret = new int[n_tags];
        for (int i = 0; i < n_tags; i++) {
            ret[i] = emitCount[i][word];
        }
        return ret;
    }

    public int[] getPrevToCurrCount(int prevTag) {
        return bigramCount[prevTag];
    }

    public int[] getCurrToNextCount(int nextTag) {
        int[] ret = new int[n_tags];
        for (int i = 0; i < n_tags; i++) {
            ret[i] = bigramCount[i][nextTag];
        }
        return ret;
    }

    public void changeBigramCount(int prevTag, int nextTag, int oldTag, int newTag) {
        if (prevTag != -1) {
            bigramCount[prevTag][oldTag]--;
            bigramCount[prevTag][newTag]++;
        }
        if (nextTag != -1) {
            bigramCount[oldTag][nextTag]--;
            bigramCount[newTag][nextTag]++;
        }
    }

    public void changeEmitCount(int word, int oldTag, int newTag) {
        emitCount[oldTag][word] --;
        emitCount[newTag][word] ++;
    }

    public void changeMonoCount(int oldTag, int newTag) {
        monoCount[oldTag] --;
        monoCount[newTag] ++;
    }

    public double getLogProb() {
        double logProb = Math.log(pi[tags[0]]);
        for (int i = 1; i < length; i++) {
            logProb += Math.log(baseTransit[tags[i - 1]][tags[i]]);
        }
        return logProb;
    }
}
