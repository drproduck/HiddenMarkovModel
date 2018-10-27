package util;

import math.CategoricalDistribution;

public class MatrixHelper {
    /**
     * equivalent to a[:,nd,rd]
     * @param a 3d array
     * @param nd index of second dimension
     * @param rd index of third dimension
     * @return a[:,nd,rd]
     */
    public static double[] firstSlice(double[][][] a, int nd, int rd){
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i][nd][rd];
        }
        return res;
    }

    /**
     * Sample index of the probability matrix
     * @param matrix: 2d probability matrix, sum of all entries = 1
     * @param cd: categorical distribution class
     * @return: the sampled row and column index
     */
    public static Pair<Integer, Integer> flattenMatrixSampleIndex(double[][] matrix, CategoricalDistribution cd){
        int nrows = matrix.length;
        int ncols = matrix[0].length;
        double[] flat = new double[nrows*ncols];
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                flat[i*ncols+j] = matrix[i][j];
            }
        }
        int idx = cd.sample(flat);
        return new Pair<Integer, Integer>(idx / ncols, idx % ncols);
    }

}
