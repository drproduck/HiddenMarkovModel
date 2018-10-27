package math;

import org.apache.commons.math3.util.FastMath;

import java.util.List;

public class Math {
    /**
     * used to normalize probabilities
     * @param arr
     * @return log sum of these log probabilities
     */
    public static double logSumExp(double[] arr) {
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            max = (max < arr[i]) ? arr[i] : max;
        }
        double s = 0;
        for (int i = 0; i < arr.length; i++) {
            s += FastMath.exp(arr[i] - max);
        }
        return max + FastMath.log(s);
    }

    public static double logSumExp(List<Double> arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (Double d : arr) {
            max = (max < d) ? d : max;
        }
        double s = 0;
        for (Double d : arr) {
            s += FastMath.exp(d - max);
        }
        if (Double.isInfinite(s)) {
            System.out.println("infinity error");
        }
        return max + FastMath.log(s);
    }

    public static void logSumExpForSoftMax(double[] inPlaceArr) {
        double max = inPlaceArr[0];
        for (int i = 0; i < inPlaceArr.length; i++) {
            max = (max < inPlaceArr[i]) ? inPlaceArr[i] : max;
        }
        double s = 0;
        for (int i = 0; i < inPlaceArr.length; i++) {
            s += FastMath.exp(inPlaceArr[i] - max);
        }
        s = max + FastMath.log(s);
        for (int i = 0; i < inPlaceArr.length; i++) {
            inPlaceArr[i] = inPlaceArr[i] - s;
        }
    }

    public static void logSumExpForSoftMax(List<Double> inPlaceArr) {
        double max = Double.NEGATIVE_INFINITY;
        for (Double d : inPlaceArr) {
            max = (max < d) ? d : max;
        }
        double s = 0;
        for (Double d : inPlaceArr) {
            s += FastMath.exp(d - max);
        }
        s = max + FastMath.log(s);
        for (int i = 0; i < inPlaceArr.size(); i++) {
            inPlaceArr.set(i, inPlaceArr.get(i) - s);
        }
    }
}
