import org.apache.commons.math3.util.FastMath;
//import sun.rmi.runtime.Log;

public class SpeedTest {
    public static void main(String[] args) throws Exception {
        int[] pt = util.plain("/home/drproduck/Documents/HMM/data/408plaincleaned");
        int[] ct = util.read408("/home/drproduck/Documents/HMM/data/408ciphercleaned");
        int nTag = 26;
        int nWord = 54;
        int len = 408;

        // fast HMM
        double[][][] trigram = util.readTrigram("/home/drproduck/Documents/HMM/data/trigram.txt");
        long start = System.nanoTime();
        FastTHMM FHMM = new FastTHMM(trigram, nTag, nWord, ct);
        long stop = System.nanoTime();
        System.out.printf("initialization time: %f\n", (stop - start) / 1e9);

        start = System.nanoTime();
        FHMM.init(123456, false);
        stop = System.nanoTime();
        System.out.printf("init time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        FHMM.alpha();
        stop = System.nanoTime();
        System.out.printf("prior time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        FHMM.logProbFromAlpha();
        stop = System.nanoTime();
        System.out.printf("logprob time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        FHMM.beta();
        stop = System.nanoTime();
        System.out.printf("beta time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        FHMM.digamma();
        stop = System.nanoTime();
        System.out.printf("digamma time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        FHMM.trigamma();
        stop = System.nanoTime();
        System.out.printf("trigamma time = %f\n", (stop - start) / 1e9);

        for (int i = 0; i < nTag; i++) {
            for (int j = 0; j < nWord; j++) {
                System.out.print(FastMath.log(FHMM.B[i][j])+" ");
            }
            System.out.println();
        }


        // log-scale HMM
        double[][][] logtrigram = util.readTrigram("/home/drproduck/Documents/HMM/data/logtrigram.txt");
        start = System.nanoTime();
        TriHMM LogHMM = new TriHMM(logtrigram, nTag, nWord, ct);
        stop = System.nanoTime();
        System.out.printf("initialization time: %f\n", (stop - start) / 1e9);

        start = System.nanoTime();
        LogHMM.init(123456, false);
        stop = System.nanoTime();
        System.out.printf("init time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        LogHMM.alpha();
        stop = System.nanoTime();
        System.out.printf("prior time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        LogHMM.logProbFromAlpha();
        stop = System.nanoTime();
        System.out.printf("logprob time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        LogHMM.beta();
        stop = System.nanoTime();
        System.out.printf("beta time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        LogHMM.digamma();
        LogHMM.gamma();
        stop = System.nanoTime();
        System.out.printf("digamma time = %f\n", (stop - start) / 1e9);
        start = System.nanoTime();
        LogHMM.trigamma();
        stop = System.nanoTime();
        System.out.printf("trigamma time = %f\n", (stop - start) / 1e9);
        for (int i = 0; i < nTag; i++) {
            for (int j = 0; j < nWord; j++) {
                System.out.print(LogHMM.B[i][j]+" ");
            }
            System.out.println();
        }

        int s = 0;
        System.out.println("\nEstimation error");
        for (int i = 0; i < nTag; i++) {
            for (int j = 0; j < nTag; j++) {
                for (int k = 0; k < nTag; k++) {
                    double e = LogHMM.A[i][j][k] - FastMath.log(FHMM.A[i][j][k]);
                    System.out.print(e + " ");
                    if (e > 1e-12) {
                        s ++;
                    }
                }
                System.out.println();
            }

        }
        System.out.printf("Intolerable entries: %d\n", s);
    }

}
