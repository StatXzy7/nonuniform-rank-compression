package com.lowrank;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

/**
 * Thin (economy) SVD of one matrix: A = U * diag(s) * Vt.
 *
 * <p>Singular values are ordered from largest to smallest. Values at or below
 * {@link LowRankEngine#ZERO_REL} times the leading value are treated as exact
 * zeros, so a numerically-rank-deficient matrix behaves like its mathematical
 * counterpart and no division by zero ever occurs.
 */
final class ThinSpectrum {
  final int m;
  final int n;
  final int k;
  /** U is m by k, Vt is k by n. */
  final double[][] u;
  final double[][] vt;
  /** Length k; exact zeros carry 0.0. */
  final double[] sigma;
  /** Normalized value sigma[i] / sigma[0]; exact zeros carry 0.0. */
  final double[] level;
  /** Whether every singular value is zero (the matrix is the zero matrix). */
  final boolean allZero;

  private ThinSpectrum(int m, int n, double[][] u, double[][] vt,
                       double[] sigma, double[] level, boolean allZero) {
    this.m = m;
    this.n = n;
    this.k = sigma.length;
    this.u = u;
    this.vt = vt;
    this.sigma = sigma;
    this.level = level;
    this.allZero = allZero;
  }

  static ThinSpectrum compute(NamedMatrix matrix) {
    int m = matrix.rows();
    int n = matrix.cols();
    SimpleMatrix a = new SimpleMatrix(m, n);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        a.set(i, j, matrix.get(i, j));
      }
    }
    @SuppressWarnings("unchecked")
    SimpleSVD<SimpleMatrix> svd = a.svd(true);
    SimpleMatrix su = svd.getU();
    SimpleMatrix sw = svd.getW();
    SimpleMatrix sv = svd.getV();
    int k = Math.min(m, n);
    double[] sigma = new double[k];
    double smax = 0.0;
    for (int i = 0; i < k; i++) {
      sigma[i] = sw.get(i, i);
      if (sigma[i] > smax) {
        smax = sigma[i];
      }
    }
    double zeroCut = LowRankEngine.ZERO_REL * smax;
    boolean allZero = true;
    double[] level = new double[k];
    for (int i = 0; i < k; i++) {
      if (sigma[i] <= zeroCut) {
        sigma[i] = 0.0;
        level[i] = 0.0;
      } else {
        allZero = false;
        level[i] = sigma[i] / smax;
      }
    }
    double[][] u = new double[m][k];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        u[i][j] = su.get(i, j);
      }
    }
    double[][] vt = new double[k][n];
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        vt[i][j] = sv.get(j, i);
      }
    }
    return new ThinSpectrum(m, n, u, vt, sigma, level, allZero);
  }
}
