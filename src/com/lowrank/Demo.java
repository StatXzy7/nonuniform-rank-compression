package com.lowrank;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Small end-to-end demonstration: build a named batch, compress it both ways,
 * print per-matrix reports, persist/reload, and multiply two reconstructions.
 */
public final class Demo {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.ROOT);

    List<NamedMatrix> batch = Arrays.asList(
        new NamedMatrix("smooth", new double[][] {
            {3.0, 0.1, 0.0},
            {0.1, 2.8, 0.1},
            {0.0, 0.1, 2.5}}),
        new NamedMatrix("spiky", new double[][] {
            {5.0, 0.2, 0.01},
            {0.2, 0.4, 0.0},
            {0.01, 0.0, 0.05}}),
        new NamedMatrix("wide", new double[][] {
            {2.0, 0.0, 0.0, 0.0},
            {0.0, 0.3, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0}}),
        new NamedMatrix("zeros", new double[2][3]),
        new NamedMatrix("pair", new double[][] {
            {1.0, 0.0},
            {0.0, 1.0}}));

    LowRankEngine engine = new LowRankEngine(batch);

    System.out.println("=== explicit threshold 0.30 ===");
    BatchResult byThreshold = engine.compressWithThreshold(0.30);
    printReport(byThreshold);

    System.out.println();
    System.out.println("=== target deletion ratio 0.34 ===");
    BatchResult byTarget = engine.compressToTargetRatio(0.34);
    printReport(byTarget);

    System.out.println();
    System.out.println("=== persistence ===");
    Path file = FilesSupport.tempPath("demo-", ".nlrm");
    byTarget.save(file);
    BatchResult reloaded = BatchResult.load(file);
    double diff = 0.0;
    for (CompressedMatrix cm : byTarget.matrices()) {
      double[][] a = cm.reconstruct();
      double[][] b = reloaded.matrix(cm.name()).reconstruct();
      for (int i = 0; i < a.length; i++) {
        for (int j = 0; j < a[i].length; j++) {
          diff = Math.max(diff, Math.abs(a[i][j] - b[i][j]));
        }
      }
    }
    System.out.printf("saved %s, reload max reconstruction diff = %.2e%n",
        file, diff);

    System.out.println();
    System.out.println("=== multiplication ===");
    NamedMatrix left = new NamedMatrix("L",
        new double[][] {{1, 2, 0}, {0, 1, 1}});
    NamedMatrix right = new NamedMatrix("R",
        new double[][] {{1, 0}, {0, 1}, {1, 1}});
    BatchResult mul = new LowRankEngine(Arrays.asList(left, right))
        .compressWithThreshold(0.0);
    double[][] product = mul.multiply("L", "R");
    System.out.println("truncated(L) * truncated(R) =");
    for (double[] row : product) {
      System.out.printf("  [%6.3f %6.3f]%n", row[0], row[1]);
    }
  }

  private static void printReport(BatchResult r) {
    if (r.isTargetMode()) {
      System.out.printf(
          "mode=%s target=%.4f reachable actual=%.4f signed deviation=%+.4f "
              + "(abs %.4f)%n",
          r.modeName(), r.requestedValue(), r.actualDeletedRatio(),
          r.deviation(), r.absoluteDeviation());
      System.out.printf("effective threshold that reproduces this cut = %.6f%n",
          r.effectiveThreshold());
    } else {
      System.out.printf("mode=%s threshold=%.4f actual deleted ratio=%.4f%n",
          r.modeName(), r.requestedValue(), r.actualDeletedRatio());
    }
    System.out.printf(
        "directions: total=%d retained=%d deleted=%d (zeros counted)%n",
        r.totalDirections(), r.retainedDirections(), r.deletedDirections());
    System.out.printf("%-8s %4s %7s %8s %9s %9s  %-22s%n",
        "name", "rank", "deleted", "relErr", "factors", "original",
        "storage form (spectral / actual)");
    for (CompressedMatrix cm : r.matrices()) {
      String form;
      if (cm.rank() == 0) {
        form = "rank 0 (empty factors) / 0 doubles";
      } else if (cm.factorHasSavings()) {
        form = "factors / factors (saves "
            + (cm.originalElementCount() - cm.factorElementCount()) + ")";
      } else {
        form = "factors / DENSE truncation (no factor gain)";
      }
      System.out.printf(Locale.ROOT,
          "%-8s %4d %7d %8.5f %9d %9d  %s%n",
          cm.name(), cm.rank(), cm.deletedDirections(), cm.relativeFroError(),
          cm.factorElementCount(), cm.originalElementCount(), form);
    }
  }

  private Demo() {
  }
}
