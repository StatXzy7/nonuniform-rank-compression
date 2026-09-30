package com.lowrank;

/**
 * Spectrally truncated result for one matrix, together with the storage
 * decision for that result.
 *
 * <p>The spectral result is always the rank-r truncation A_r = U diag(s) Vt.
 * Two concrete storage forms are reported separately:
 * <ul>
 *   <li>factor form: U (m*r), s (r), Vt (r*n) &mdash; r*(m+n+1) doubles;</li>
 *   <li>dense form: the truncated matrix itself &mdash; m*n doubles.</li>
 * </ul>
 * The in-memory object always retains both views (they are tiny, dims &le; 12);
 * only the persistent snapshot stores the chosen form.
 */
public final class CompressedMatrix {
  private final String name;
  private final int rows;
  private final int cols;
  private final int rank;
  private final double[][] truncated;
  private final double[][] u;
  private final double[] s;
  private final double[][] vt;
  private final double relFroError;
  private final boolean storedDense;

  CompressedMatrix(String name, ThinSpectrum spec, int rank, double[][] truncated,
                   double[][] u, double[] s, double[][] vt,
                   double relFroError, boolean storedDense) {
    this(name, spec.m, spec.n, rank, truncated, u, s, vt, relFroError, storedDense);
  }

  CompressedMatrix(String name, int rows, int cols, int rank, double[][] truncated,
                   double[][] u, double[] s, double[][] vt,
                   double relFroError, boolean storedDense) {
    this.name = name;
    this.rows = rows;
    this.cols = cols;
    this.rank = rank;
    this.truncated = truncated;
    this.u = u;
    this.s = s;
    this.vt = vt;
    this.relFroError = relFroError;
    this.storedDense = storedDense;
  }

  public String name() {
    return name;
  }

  public int rows() {
    return rows;
  }

  public int cols() {
    return cols;
  }

  /** Retained rank after spectral truncation (0 for an all-zero matrix). */
  public int rank() {
    return rank;
  }

  /** Number of thin-spectrum directions, including deleted zero directions. */
  public int thinDirections() {
    return Math.min(rows, cols);
  }

  public int deletedDirections() {
    return thinDirections() - rank;
  }

  /** Relative Frobenius reconstruction error ||A-A_r||_F / ||A||_F (0 for A=0). */
  public double relativeFroError() {
    return relFroError;
  }

  /** Number of doubles needed to store the low-rank factors: r*(m+n+1). */
  public int factorElementCount() {
    return rank * (rows + cols + 1);
  }

  /** Number of doubles in the original dense matrix: m*n. */
  public int originalElementCount() {
    return rows * cols;
  }

  /** Number of doubles in the chosen storage form. */
  public int storedElementCount() {
    return storedDense ? originalElementCount() : factorElementCount();
  }

  /** True iff the factor form uses strictly fewer doubles than the dense form. */
  public boolean factorHasSavings() {
    return factorElementCount() < originalElementCount();
  }

  /** Whether the snapshot keeps a dense truncated matrix instead of factors. */
  public boolean storedAsDense() {
    return storedDense;
  }

  /** True iff the chosen storage form is smaller than the original matrix. */
  public boolean storageSaves() {
    return storedElementCount() < originalElementCount();
  }

  /** Rebuilds the truncated matrix A_r (defensive copy). */
  public double[][] reconstruct() {
    double[][] out = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(truncated[i], 0, out[i], 0, cols);
    }
    return out;
  }

  double[][] truncatedArray() {
    return truncated;
  }

  double[][] factorU() {
    return u;
  }

  double[] factorS() {
    return s;
  }

  double[][] factorVt() {
    return vt;
  }

  /**
   * Rebuilds A_r directly from the factors (with a fixed summation order),
   * used to prove the saved factors reconstruct the truncated result.
   */
  static double[][] reconstructFromFactors(int m, int n, int rank,
                                           double[][] u, double[] s, double[][] vt) {
    double[][] out = new double[m][n];
    for (int r = 0; r < rank; r++) {
      double sr = s[r];
      for (int i = 0; i < m; i++) {
        double coef = sr * u[i][r];
        double[] row = out[i];
        double[] vr = vt[r];
        for (int j = 0; j < n; j++) {
          row[j] += coef * vr[j];
        }
      }
    }
    return out;
  }

  /** Plain double matrix multiplication, shared by pre/post-reload results. */
  static double[][] multiply(double[][] a, double[][] b) {
    int m = a.length;
    int k = b.length;
    int n = b[0].length;
    double[][] out = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int p = 0; p < k; p++) {
        double aip = a[i][p];
        for (int j = 0; j < n; j++) {
          out[i][j] += aip * b[p][j];
        }
      }
    }
    return out;
  }
}
