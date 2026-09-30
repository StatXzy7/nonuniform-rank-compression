package com.lowrank;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

/**
 * Non-uniform low-rank compression engine.
 *
 * <p>All matrices in a batch share one direction-setting threshold, but each
 * matrix normalizes its own singular values by its own largest singular value,
 * so different spectra naturally keep different ranks. A nonzero direction
 * survives when its normalized singular value is at or above the threshold;
 * exact equality is retained, so equal normalized values always move together
 * and are never split to chase a budget. Zero singular values are always
 * deleted.
 *
 * <p>Positive normalized values are clustered (within {@link #LEVEL_EPS}) into
 * strictly increasing spectral levels. The reachable cuts are prefixes of the
 * level list, so a batch offers only a discrete set of deletion ratios; the
 * denominator is the total number of thin-spectrum directions, zeros included.
 */
public final class LowRankEngine {
  /** A singular value at or below this fraction of the max is treated as zero. */
  public static final double ZERO_REL = 1.0e-12;
  /** Normalized values within this absolute tolerance share one spectral level. */
  public static final double LEVEL_EPS = 1.0e-9;

  private static final class Entry {
    final NamedMatrix matrix;
    final ThinSpectrum spec;
    /** Cluster index per thin direction; -1 for exact zeros. */
    final int[] cluster;

    Entry(NamedMatrix matrix, ThinSpectrum spec, int[] cluster) {
      this.matrix = matrix;
      this.spec = spec;
      this.cluster = cluster;
    }
  }

  private final List<Entry> entries;
  /** Strictly increasing distinct positive normalized spectral levels. */
  private final double[] levels;
  private final int totalDirections;

  public LowRankEngine(List<NamedMatrix> matrices) {
    if (matrices == null || matrices.isEmpty()) {
      throw new CompressionException("batch must contain at least one matrix");
    }
    Set<String> names = new HashSet<>();
    List<Entry> built = new ArrayList<>();
    int dirs = 0;
    for (NamedMatrix matrix : matrices) {
      if (matrix == null) {
        throw new CompressionException("batch contains a null matrix");
      }
      if (!names.add(matrix.name())) {
        throw new CompressionException("duplicate matrix name: '" + matrix.name() + "'");
      }
      ThinSpectrum spec = ThinSpectrum.compute(matrix);
      built.add(new Entry(matrix, spec, new int[spec.k]));
      dirs += spec.k;
    }
    this.entries = Collections.unmodifiableList(built);
    this.totalDirections = dirs;
    this.levels = collectLevels(built);
    assignClusters();
  }

  private static double[] collectLevels(List<Entry> entries) {
    List<Double> positive = new ArrayList<>();
    for (Entry e : entries) {
      for (double v : e.spec.level) {
        if (v > 0.0) {
          positive.add(v);
        }
      }
    }
    Collections.sort(positive);
    List<Double> clustered = new ArrayList<>();
    for (double v : positive) {
      if (clustered.isEmpty()
          || v - clustered.get(clustered.size() - 1) > LEVEL_EPS) {
        clustered.add(v);
      }
    }
    double[] out = new double[clustered.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = clustered.get(i);
    }
    return out;
  }

  private void assignClusters() {
    for (Entry e : entries) {
      for (int i = 0; i < e.spec.k; i++) {
        double v = e.spec.level[i];
        if (v == 0.0) {
          e.cluster[i] = -1;
          continue;
        }
        int idx = -1;
        for (int c = 0; c < levels.length; c++) {
          if (Math.abs(v - levels[c]) <= LEVEL_EPS) {
            idx = c;
            break;
          }
        }
        if (idx < 0) {
          throw new IllegalStateException("internal: unclustered level " + v);
        }
        e.cluster[i] = idx;
      }
    }
  }

  /**
   * Number of deleted directions when retaining clusters {@code cut..L-1}.
   * Zero directions are always counted as deleted.
   */
  private int deletedCount(int cut) {
    int deleted = 0;
    for (Entry e : entries) {
      for (int c : e.cluster) {
        if (c < 0 || c < cut) {
          deleted++;
        }
      }
    }
    return deleted;
  }

  /**
   * Reachable cut indices 0..{@code reachableCutCount()-1}; a cut is reachable
   * only if some threshold in [0,1] produces it. Dropping the largest level is
   * impossible when it equals 1.0, because nonzero values equal to the
   * threshold are always retained.
   */
  private int reachableCutCount() {
    int l = levels.length;
    if (l == 0) {
      return 1;
    }
    // Each matrix's top normalized value is exactly 1.0, so the top level of
    // any batch containing a nonzero matrix equals 1 and cannot be dropped.
    if (levels[l - 1] < 1.0 - LEVEL_EPS) {
      return l + 1;
    }
    return l;
  }

  private double ratioAt(int cut) {
    return (double) deletedCount(cut) / totalDirections;
  }

  private static double checkUnitRange(double value, String what) {
    if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
      throw new CompressionException(what + " must lie in [0,1], got " + value);
    }
    return value;
  }

  /** Compresses the batch with an explicit shared threshold in [0,1]. */
  public BatchResult compressWithThreshold(double threshold) {
    checkUnitRange(threshold, "threshold");
    int cut = 0;
    while (cut < levels.length && levels[cut] < threshold - LEVEL_EPS) {
      cut++;
    }
    return build(BatchResult.Mode.THRESHOLD, threshold, cut);
  }

  /**
   * Compresses the batch toward a target deletion ratio in [0,1]. The engine
   * picks the reachable cut whose realized ratio is closest; exact ties keep
   * more directions (the smaller cut). Equal spectral values are never split.
   */
  public BatchResult compressToTargetRatio(double target) {
    checkUnitRange(target, "target deletion ratio");
    int cuts = reachableCutCount();
    int best = 0;
    double bestGap = Math.abs(ratioAt(0) - target);
    for (int cut = 1; cut < cuts; cut++) {
      double gap = Math.abs(ratioAt(cut) - target);
      if (gap < bestGap - 1.0e-15) {
        bestGap = gap;
        best = cut;
      }
    }
    return build(BatchResult.Mode.TARGET_RATIO, target, best);
  }

  /**
   * Canonical threshold that reproduces {@code cut} through the public
   * threshold path. Since equality is retained, the reproducing value sits
   * strictly between the highest dropped level and the lowest retained level
   * (their midpoint); 0 means no deletion and 1 is used only when every
   * positive level is below 1.
   */
  private double canonicalThreshold(int cut) {
    if (cut == 0) {
      return 0.0;
    }
    if (cut < levels.length) {
      return 0.5 * (levels[cut - 1] + levels[cut]);
    }
    return 1.0;
  }

  private BatchResult build(BatchResult.Mode mode, double requested, int cut) {
    LinkedHashMap<String, CompressedMatrix> out = new LinkedHashMap<>();
    for (Entry e : entries) {
      out.put(e.matrix.name(), truncate(e.matrix.name(), e.spec, e.cluster, cut));
    }
    return new BatchResult(mode, requested, canonicalThreshold(cut),
        totalDirections, deletedCount(cut), out);
  }

  private static CompressedMatrix truncate(String name, ThinSpectrum spec,
                                           int[] cluster, int cut) {
    List<Integer> kept = new ArrayList<>();
    for (int i = 0; i < spec.k; i++) {
      if (cluster[i] >= cut) {
        kept.add(i);
      }
    }
    int rank = kept.size();
    double[][] u = new double[spec.m][rank];
    double[] s = new double[rank];
    double[][] vt = new double[rank][spec.n];
    for (int r = 0; r < rank; r++) {
      int idx = kept.get(r);
      s[r] = spec.sigma[idx];
      for (int i = 0; i < spec.m; i++) {
        u[i][r] = spec.u[i][idx];
      }
      System.arraycopy(spec.vt[idx], 0, vt[r], 0, spec.n);
    }
    double[][] truncated =
        CompressedMatrix.reconstructFromFactors(spec.m, spec.n, rank, u, s, vt);

    double relError = 0.0;
    if (!spec.allZero) {
      double normSq = 0.0;
      double errSq = 0.0;
      boolean[] isKept = new boolean[spec.k];
      for (int idx : kept) {
        isKept[idx] = true;
      }
      for (int i = 0; i < spec.k; i++) {
        double sv = spec.sigma[i];
        normSq += sv * sv;
        if (!isKept[i]) {
          errSq += sv * sv;
        }
      }
      relError = Math.sqrt(errSq / normSq);
    }
    int factorCount = rank * (spec.m + spec.n + 1);
    boolean dense = factorCount >= spec.m * spec.n;
    return new CompressedMatrix(name, spec, rank, truncated, u, s, vt,
        relError, dense);
  }
}
