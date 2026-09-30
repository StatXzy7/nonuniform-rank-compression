package com.lowrank;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Automated behavior tests for the compression engine. */
public final class EngineTest {
  private static final double TOL = 1.0e-12;

  public static void main(String[] args) throws Exception {
    inputValidation();
    differentSpectraDifferentRanks();
    thresholdEqualityKeptAndZeroDropped();
    discreteBudgetAndTieKeepsMore();
    equalValuesNeverSplit();
    zeroMatrixRankZero();
    zerosCountInBatchDenominator();
    reconstructionErrors();
    denseWhenFactorsGiveNoSavings();
    saveLoadRoundTrip();
    denseSnapshotRoundTrip();
    corruptSnapshotsRejected();
    failedSaveIsAtomic();
    multiplyConsistentAcrossReload();
    signAndRegaugingIndependence();
    nonUniformPerMatrixRanks();
    targetModeReporting();
    maxLevelOneCannotDeleteIt();
    System.out.println();
    System.out.println("ALL TESTS PASSED");
  }

  // ---- helpers ----------------------------------------------------------

  static NamedMatrix diag(String name, double... vals) {
    int k = vals.length;
    double[][] a = new double[k][k];
    for (int i = 0; i < k; i++) {
      a[i][i] = vals[i];
    }
    return new NamedMatrix(name, a);
  }

  static NamedMatrix rectangularDiag(String name, int rows, int cols,
                                     double... vals) {
    double[][] a = new double[rows][cols];
    for (int i = 0; i < vals.length; i++) {
      a[i][i] = vals[i];
    }
    return new NamedMatrix(name, a);
  }

  static double[][] mat(double[]... rows) {
    double[][] out = new double[rows.length][];
    for (int i = 0; i < rows.length; i++) {
      out[i] = rows[i].clone();
    }
    return out;
  }

  static List<NamedMatrix> list(NamedMatrix... ms) {
    return new ArrayList<>(Arrays.asList(ms));
  }

  static void assertMatrixClose(TestHarness h, double[][] actual,
                                double[][] expected, double tol, String msg) {
    h.checkEq(actual.length, expected.length, msg + " rows");
    for (int i = 0; i < expected.length; i++) {
      h.checkEq(actual[i].length, expected[i].length, msg + " cols row " + i);
      for (int j = 0; j < expected[i].length; j++) {
        h.checkClose(actual[i][j], expected[i][j], tol,
            msg + " at [" + i + "," + j + "]");
      }
    }
  }

  static double maxAbsDiff(double[][] a, double[][] b) {
    double max = 0.0;
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < a[i].length; j++) {
        max = Math.max(max, Math.abs(a[i][j] - b[i][j]));
      }
    }
    return max;
  }

  static Path tempFile(String suffix) throws IOException {
    Path p = Files.createTempFile("nlrc-", suffix);
    Files.deleteIfExists(p);
    return p;
  }

  // ---- tests ------------------------------------------------------------

  static void inputValidation() {
    TestHarness.run("invalid inputs rejected", h -> {
      h.expectThrows(CompressionException.class,
          () -> new LowRankEngine(new ArrayList<>()), "empty batch rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[0][]), "empty rows rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[][] {{}}), "empty cols rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[13][1]), "13 rows rejected");
      double[][] big = new double[1][13];
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", big), "13 cols rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("", new double[][] {{1}}), "blank name rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[][] {{Double.NaN}}),
          "NaN rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[][] {{1, Double.POSITIVE_INFINITY}}),
          "infinity rejected");
      h.expectThrows(CompressionException.class,
          () -> new NamedMatrix("x", new double[][] {{1, 2}, {3}}),
          "ragged matrix rejected");
      NamedMatrix one = diag("dup", 1);
      h.expectThrows(CompressionException.class,
          () -> new LowRankEngine(list(one, diag("dup", 2))),
          "duplicate names rejected");
      LowRankEngine engine = new LowRankEngine(list(diag("a", 1)));
      h.expectThrows(CompressionException.class,
          () -> engine.compressWithThreshold(-0.01), "negative threshold rejected");
      h.expectThrows(CompressionException.class,
          () -> engine.compressWithThreshold(1.01), "threshold > 1 rejected");
      h.expectThrows(CompressionException.class,
          () -> engine.compressWithThreshold(Double.NaN), "NaN threshold rejected");
      h.expectThrows(CompressionException.class,
          () -> engine.compressToTargetRatio(-0.5), "negative target rejected");
      h.expectThrows(CompressionException.class,
          () -> engine.compressToTargetRatio(1.5), "target > 1 rejected");
    });
  }

  static void differentSpectraDifferentRanks() {
    TestHarness.run("different spectra keep different ranks", h -> {
      NamedMatrix slow = diag("slow", 1.0, 0.8, 0.6);
      NamedMatrix fast = diag("fast", 1.0, 0.2, 0.05);
      BatchResult r = new LowRankEngine(list(slow, fast))
          .compressWithThreshold(0.5);
      h.checkEq(r.matrix("slow").rank(), 3, "flat spectrum keeps rank 3");
      h.checkEq(r.matrix("fast").rank(), 1, "decaying spectrum keeps rank 1");
      h.checkEq(r.totalDirections(), 6, "denominator counts all thin dirs");
      h.checkEq(r.deletedDirections(), 2, "two directions deleted");
      h.checkClose(r.actualDeletedRatio(), 2.0 / 6.0, TOL, "ratio over all dirs");
    });
  }

  static void thresholdEqualityKeptAndZeroDropped() {
    TestHarness.run("equality retained and zeros always dropped", h -> {
      NamedMatrix m = diag("eq", 1.0, 0.5, 0.25);
      BatchResult r = new LowRankEngine(list(m)).compressWithThreshold(0.5);
      h.checkEq(r.matrix("eq").rank(), 2, "sigma == threshold is retained");
      BatchResult rZero = new LowRankEngine(list(m)).compressWithThreshold(0.0);
      h.checkEq(rZero.matrix("eq").rank(), 3, "threshold 0 keeps all nonzero");
    });
  }

  static void discreteBudgetAndTieKeepsMore() {
    TestHarness.run("discrete reachable budget, tie keeps more", h -> {
      // Each 2x2 matrix has two distinct normalized levels (1.0, ~0.3); the
      // only batch cuts keep 0, 2 or 4 of 4 directions -> ratios 0, .5, 1.
      double v = 0.3;
      BatchResult r = new LowRankEngine(list(diag("a", 1, v), diag("b", 1, v)))
          .compressToTargetRatio(0.25);
      h.checkEq(r.deletedDirections(), 0, "tie at .25 prefers deleting fewer");
      h.checkClose(r.actualDeletedRatio(), 0.0, TOL, "actual ratio is 0");
      h.checkClose(r.absoluteDeviation(), 0.25, TOL, "deviation reported");
      BatchResult r2 = new LowRankEngine(list(diag("a", 1, v), diag("b", 1, v)))
          .compressToTargetRatio(0.3);
      h.checkEq(r2.deletedDirections(), 2, ".30 is closer to .50 than 0");
      h.checkClose(r2.actualDeletedRatio(), 0.5, TOL, "actual ratio is .5");
    });
  }

  static void equalValuesNeverSplit() {
    TestHarness.run("equal spectral values are never split", h -> {
      NamedMatrix m = diag("eq", 1.0, 0.5, 0.5);
      BatchResult r = new LowRankEngine(list(m)).compressToTargetRatio(1.0 / 3.0);
      // Reachable ratios are 0/3 and 2/3 (both 0.5s share one level); the
      // engine must not keep 2/3 rank by splitting the equal pair.
      boolean valid = r.matrix("eq").rank() == 3 || r.matrix("eq").rank() == 1;
      h.check(valid, "rank at nearest budget never splits equal values (rank="
          + r.matrix("eq").rank() + ")");
      h.checkEq(r.matrix("eq").rank(), 3, "tie between 0/3 and 2/3 keeps more");
    });
  }

  static void zeroMatrixRankZero() {
    TestHarness.run("all-zero matrix stays rank zero", h -> {
      NamedMatrix z = new NamedMatrix("zero", new double[3][4]);
      BatchResult r = new LowRankEngine(list(z)).compressWithThreshold(0.0);
      CompressedMatrix cm = r.matrix("zero");
      h.checkEq(cm.rank(), 0, "zero matrix rank is 0");
      h.checkEq(cm.thinDirections(), 3, "thin directions = min(3,4)");
      h.checkEq(cm.deletedDirections(), 3, "all directions deleted");
      h.checkEq(cm.factorElementCount(), 0, "zero-rank factors hold no doubles");
      h.checkClose(cm.relativeFroError(), 0.0, TOL, "no division by zero / error 0");
      assertMatrixClose(h, cm.reconstruct(), new double[3][4], 0.0,
          "zero reconstruction");
      h.checkClose(r.actualDeletedRatio(), 1.0, TOL, "whole batch deleted");
      BatchResult rt = new LowRankEngine(list(z)).compressToTargetRatio(0.9);
      h.checkEq(rt.matrix("zero").rank(), 0, "target mode also rank 0");
    });
  }

  static void zerosCountInBatchDenominator() {
    TestHarness.run("zero directions join the shared denominator", h -> {
      NamedMatrix z = new NamedMatrix("zero", new double[2][2]);
      NamedMatrix full = diag("full", 1, 1);
      BatchResult r = new LowRankEngine(list(z, full)).compressWithThreshold(0.0);
      h.checkEq(r.totalDirections(), 4, "2 + 2 thin directions");
      h.checkEq(r.deletedDirections(), 2, "only the zero directions are deleted");
      h.checkClose(r.actualDeletedRatio(), 0.5, TOL, "denominator includes zeros");
    });
  }

  static void reconstructionErrors() {
    TestHarness.run("reconstruction matches the truncated spectrum", h -> {
      double[] sig = {3.0, 2.0, 1.0};
      NamedMatrix m = diag("d", sig);
      BatchResult r = new LowRankEngine(list(m)).compressWithThreshold(0.6);
      CompressedMatrix cm = r.matrix("d");
      h.checkEq(cm.rank(), 2, "two values above 0.6 (1.0 and 0.667)");
      double expected = 1.0 / Math.sqrt(9 + 4 + 1);
      h.checkClose(cm.relativeFroError(), expected, 1.0e-14,
          "relative Frobenius error from dropped energy");
      double[][] want = {{3, 0, 0}, {0, 2, 0}, {0, 0, 0}};
      assertMatrixClose(h, cm.reconstruct(), want, 1.0e-13, "truncated matrix");
      // Same check on a generic (rotated) matrix: error is ||A-Ar||/||A||.
      double[][] dense = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
      NamedMatrix g = new NamedMatrix("g", dense);
      BatchResult rg = new LowRankEngine(list(g)).compressWithThreshold(0.2);
      CompressedMatrix cg = rg.matrix("g");
      double direct = maxDiffDirect(dense, cg.reconstruct()) / normF(dense);
      h.checkClose(cg.relativeFroError(), direct, 1.0e-12,
          "reported error equals direct residual for rotated matrix");
    });
  }

  private static double maxDiffDirect(double[][] a, double[][] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < a[i].length; j++) {
        double d = a[i][j] - b[i][j];
        sum += d * d;
      }
    }
    return Math.sqrt(sum);
  }

  private static double normF(double[][] a) {
    double sum = 0.0;
    for (double[] row : a) {
      for (double v : row) {
        sum += v * v;
      }
    }
    return Math.sqrt(sum);
  }

  static void denseWhenFactorsGiveNoSavings() {
    TestHarness.run("dense storage when factors do not pay off", h -> {
      // Full-rank small square: factors cost n(2n+1) > n*n.
      NamedMatrix full = diag("full", 4, 3, 2);
      BatchResult r = new LowRankEngine(list(full)).compressWithThreshold(0.0);
      CompressedMatrix cm = r.matrix("full");
      h.checkEq(cm.rank(), 3, "spectral result keeps rank 3");
      h.check(!cm.factorHasSavings(), "3(3+3+1)=21 > 9 dense elements");
      h.checkEq(cm.factorElementCount(), 21, "factor element count reported");
      h.checkEq(cm.originalElementCount(), 9, "original element count reported");
      h.check(cm.storedAsDense(), "snapshot falls back to dense");
      h.checkEq(cm.storedElementCount(), 9, "stored count is dense count");
      h.check(!cm.storageSaves(), "no real storage gain at full rank");

      // Rank-1 wide matrix: factors 1(m+n+1) are cheaper than m*n.
      NamedMatrix wide = rectangularDiag("wide", 3, 4, 5, 0, 0);
      BatchResult rw = new LowRankEngine(list(wide)).compressWithThreshold(0.5);
      CompressedMatrix cw = rw.matrix("wide");
      h.checkEq(cw.rank(), 1, "rank 1");
      h.check(cw.factorHasSavings(), "1*(3+4+1)=8 < 12 dense elements");
      h.check(!cw.storedAsDense(), "snapshot keeps factors");
      h.check(cw.storageSaves(), "real storage gain reported");
    });
  }

  static void saveLoadRoundTrip() {
    TestHarness.run("save/load preserves factors, reports and control values", h -> {
      try {
        NamedMatrix a = diag("a", 5, 1, 0.1);
        NamedMatrix b = rectangularDiag("b", 2, 3, 2, 1);
        BatchResult r = new LowRankEngine(list(a, b)).compressToTargetRatio(0.4);
        Path p = tempFile(".nlrm");
        r.save(p);
        BatchResult loaded = BatchResult.load(p);
        compareResults(h, r, loaded, 1.0e-14);
        Files.deleteIfExists(p);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  static void denseSnapshotRoundTrip() {
    TestHarness.run("dense snapshot reloads the same truncated matrix", h -> {
      try {
        NamedMatrix full = new NamedMatrix("full",
            new double[][] {{4, 1, 0}, {1, 3, 1}, {0, 1, 2}});
        BatchResult r = new LowRankEngine(list(full)).compressWithThreshold(0.0);
        CompressedMatrix cm = r.matrix("full");
        h.check(cm.storedAsDense(), "full-rank snapshot stored dense");
        Path p = tempFile(".nlrm");
        r.save(p);
        BatchResult loaded = BatchResult.load(p);
        compareResults(h, r, loaded, 1.0e-14);
        Files.deleteIfExists(p);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  static void corruptSnapshotsRejected() {
    TestHarness.run("truncated or tampered snapshots are rejected", h -> {
      try {
        BatchResult good = new LowRankEngine(list(diag("a", 3, 1, 0.1)))
            .compressWithThreshold(0.5);
        Path p = tempFile(".nlrm");
        good.save(p);
        byte[] bytes = Files.readAllBytes(p);

        byte[] truncated = Arrays.copyOf(bytes, bytes.length / 2);
        Path t1 = tempFile(".bad");
        Files.write(t1, truncated);
        h.expectThrows(CompressionException.class,
            () -> rethrowIo(() -> BatchResult.load(t1)), "truncated file rejected");

        byte[] tampered = bytes.clone();
        tampered[tampered.length - 3] ^= (byte) 0xFF;
        Path t2 = tempFile(".bad");
        Files.write(t2, tampered);
        h.expectThrows(CompressionException.class,
            () -> rethrowIo(() -> BatchResult.load(t2)), "tampered payload rejected");

        Path t3 = tempFile(".bad");
        Files.write(t3, new byte[] {'N', 'O', 'P', 'E', 0, 0, 0, 1});
        h.expectThrows(CompressionException.class,
            () -> rethrowIo(() -> BatchResult.load(t3)), "bad magic rejected");

        Files.deleteIfExists(p);
        Files.deleteIfExists(t1);
        Files.deleteIfExists(t2);
        Files.deleteIfExists(t3);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  static void failedSaveIsAtomic() {
    TestHarness.run("failed save never overwrites an intact result", h -> {
      try {
        BatchResult first = new LowRankEngine(list(diag("first", 9, 1, 0.1)))
            .compressWithThreshold(0.5);
        Path p = tempFile(".nlrm");
        first.save(p);
        byte[] before = Files.readAllBytes(p);

        BatchResult second = new LowRankEngine(list(diag("second", 7, 7, 1)))
            .compressWithThreshold(0.5);
        OutputStream failing = new OutputStream() {
          private int written = 0;
          @Override public void write(int b) throws IOException {
            if (written++ >= 8) {
              throw new IOException("simulated disk failure");
            }
          }
          @Override public void write(byte[] b, int off, int len) throws IOException {
            for (int i = 0; i < len; i++) {
              write(b[off + i]);
            }
          }
        };
        h.expectThrowsAny(IOException.class,
            () -> ResultStoreTestSupport.saveFailing(second, p, failing),
            "failing save propagates IOException");
        byte[] after = Files.readAllBytes(p);
        h.check(Arrays.equals(before, after), "intact result left byte-for-byte");
        BatchResult reloaded = BatchResult.load(p);
        h.checkEq(reloaded.matrix("first").rank(), first.matrix("first").rank(),
            "reloaded file is still the original batch");
        Files.deleteIfExists(p);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  static void multiplyConsistentAcrossReload() {
    TestHarness.run("multiply results match before and after reload", h -> {
      try {
        NamedMatrix a = new NamedMatrix("a",
            mat(new double[] {1, 2, 0}, new double[] {0, 1, 1}));
        NamedMatrix b = new NamedMatrix("b",
            mat(new double[] {1, 0}, new double[] {0, 1}, new double[] {1, 1}));
        BatchResult r = new LowRankEngine(list(a, b)).compressWithThreshold(0.0);
        double[][] before = r.multiply("a", "b");
        double[][] expect = {{1, 2}, {1, 2}};
        assertMatrixClose(h, before, expect, 1.0e-13, "A*B computed");

        Path p = tempFile(".nlrm");
        r.save(p);
        BatchResult loaded = BatchResult.load(p);
        double[][] after = loaded.multiply("a", "b");
        assertMatrixClose(h, after, before, 1.0e-14,
            "A*B identical after reload");
        NamedMatrix c = new NamedMatrix("c", mat(new double[] {1, 2, 3}));
        BatchResult rc = new LowRankEngine(list(a, b, c)).compressWithThreshold(0.0);
        h.expectThrows(CompressionException.class,
            () -> rc.multiply("a", "c"),
            "2x3 by 1x3 inner-dimension mismatch rejected");
        Files.deleteIfExists(p);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  static void signAndRegaugingIndependence() {
    TestHarness.run("result does not depend on singular-vector signs", h -> {
      // Multiplying columns/rows by -1 flips singular vector signs and rotates
      // gauge but leaves the spectrum and the truncated matrix invariant.
      double[][] base = {{2, 1, 0}, {1, 2, 1}, {0, 1, 2}};
      NamedMatrix x = new NamedMatrix("x", base);
      double[][] flipped = new double[3][3];
      int[] rowSign = {1, -1, 1};
      int[] colSign = {-1, 1, -1};
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          flipped[i][j] = rowSign[i] * base[i][j] * colSign[j];
        }
      }
      NamedMatrix y = new NamedMatrix("y", flipped);
      BatchResult rx = new LowRankEngine(list(x)).compressWithThreshold(0.4);
      BatchResult ry = new LowRankEngine(list(y)).compressWithThreshold(0.4);
      CompressedMatrix cx = rx.matrix("x");
      CompressedMatrix cy = ry.matrix("y");
      h.checkEq(cx.rank(), cy.rank(), "same rank under sign flips");
      h.checkClose(cx.relativeFroError(), cy.relativeFroError(), 1.0e-14,
          "same error under sign flips");
      double[][] tx = cx.reconstruct();
      double[][] ty = cy.reconstruct();
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          h.checkClose(tx[i][j], rowSign[i] * ty[i][j] * colSign[j], 1.0e-12,
              "truncated matrix covariant under sign flips");
        }
      }
      // Running the same batch twice gives bit-identical output.
      BatchResult again = new LowRankEngine(list(x)).compressWithThreshold(0.4);
      h.check(maxAbsDiff(again.matrix("x").reconstruct(), tx) == 0.0,
          "deterministic bit-identical repeat");
    });
  }

  static void nonUniformPerMatrixRanks() {
    TestHarness.run("one threshold yields a spread of per-matrix ranks", h -> {
      NamedMatrix r3 = diag("r3", 1, 0.9, 0.8);
      NamedMatrix r2 = diag("r2", 1, 0.5, 0.1);
      NamedMatrix r1 = diag("r1", 1, 0.05, 0.02);
      BatchResult r = new LowRankEngine(list(r3, r2, r1))
          .compressWithThreshold(0.3);
      h.checkEq(r.matrix("r3").rank(), 3, "rank 3 matrix");
      h.checkEq(r.matrix("r2").rank(), 2, "rank 2 matrix");
      h.checkEq(r.matrix("r1").rank(), 1, "rank 1 matrix");
      h.checkEq(r.deletedDirections(), 3, "deleted 0+1+2 directions");
    });
  }

  static void targetModeReporting() {
    TestHarness.run("target/actual/deviation reported in target mode", h -> {
      NamedMatrix m = diag("m", 1, 0.2);
      BatchResult r = new LowRankEngine(list(m)).compressToTargetRatio(0.7);
      h.checkEq(r.deletedDirections(), 1, "nearest reachable ratio is 1/2");
      h.checkClose(r.requestedValue(), 0.7, TOL, "target stored");
      h.checkClose(r.actualDeletedRatio(), 0.5, TOL, "actual stored");
      h.checkClose(r.deviation(), -0.2, TOL, "signed deviation actual-target");
      h.checkClose(r.absoluteDeviation(), 0.2, TOL, "absolute deviation");
      h.checkClose(r.effectiveThreshold(), 0.6, TOL,
          "canonical threshold sits between dropped and retained levels");
      BatchResult same = new LowRankEngine(list(m))
          .compressWithThreshold(r.effectiveThreshold());
      h.checkEq(same.deletedDirections(), r.deletedDirections(),
          "effective threshold reproduces cut");
    });
  }

  static void maxLevelOneCannotDeleteIt() {
    TestHarness.run("top level equal to 1 survives every threshold", h -> {
      NamedMatrix m = diag("m", 1.0, 0.01);
      LowRankEngine engine = new LowRankEngine(list(m));
      BatchResult t = engine.compressWithThreshold(1.0);
      h.checkEq(t.matrix("m").rank(), 1, "threshold 1 retains sigma == 1");
      BatchResult r = engine.compressToTargetRatio(1.0);
      h.checkEq(r.deletedDirections(), 1, "closest reachable cut keeps top level");
      h.checkClose(r.actualDeletedRatio(), 0.5, TOL, "ratio cannot reach 1 here");
      // Per-matrix normalization always puts the top level at exactly 1, so a
      // batch with any nonzero matrix can never delete every direction; only
      // an all-zero batch reaches ratio 1.0.
      NamedMatrix z = new NamedMatrix("z", new double[2][2]);
      BatchResult mixed = new LowRankEngine(list(m, z)).compressToTargetRatio(1.0);
      h.checkEq(mixed.matrix("m").rank(), 1, "nonzero top direction survives");
      h.checkEq(mixed.matrix("z").rank(), 0, "zero directions all deleted");
      h.check(mixed.actualDeletedRatio() < 1.0, "mixed batch ratio below 1");
      BatchResult onlyZero = new LowRankEngine(list(z)).compressToTargetRatio(1.0);
      h.checkClose(onlyZero.actualDeletedRatio(), 1.0, TOL,
          "all-zero batch reaches ratio 1");
      h.checkClose(onlyZero.effectiveThreshold(), 0.0, TOL,
          "all-zero batch needs no positive threshold (0.0)");
    });
  }

  // ---- shared assertions -----------------------------------------------

  static void compareResults(TestHarness h, BatchResult a, BatchResult b,
                             double tol) {
    h.check(a.modeName().equals(b.modeName()), "mode preserved");
    h.checkClose(a.requestedValue(), b.requestedValue(), tol, "requested");
    h.checkClose(a.effectiveThreshold(), b.effectiveThreshold(), tol,
        "effective threshold");
    h.checkEq(a.totalDirections(), b.totalDirections(), "total directions");
    h.checkEq(a.deletedDirections(), b.deletedDirections(), "deleted directions");
    h.checkClose(a.actualDeletedRatio(), b.actualDeletedRatio(), tol, "ratio");
    h.checkClose(a.deviation(), b.deviation(), tol, "deviation");
    h.checkEq(a.matrices().size(), b.matrices().size(), "matrix count");
    for (CompressedMatrix ca : a.matrices()) {
      CompressedMatrix cb = b.matrix(ca.name());
      h.checkEq(ca.rows(), cb.rows(), ca.name() + " rows");
      h.checkEq(ca.cols(), cb.cols(), ca.name() + " cols");
      h.checkEq(ca.rank(), cb.rank(), ca.name() + " rank");
      h.check(ca.storedAsDense() == cb.storedAsDense(), ca.name() + " storage");
      h.checkClose(ca.relativeFroError(), cb.relativeFroError(), tol,
          ca.name() + " error");
      h.checkEq(ca.factorElementCount(), cb.factorElementCount(),
          ca.name() + " factor count");
      h.checkEq(ca.originalElementCount(), cb.originalElementCount(),
          ca.name() + " original count");
      assertMatrixClose(h, cb.reconstruct(), ca.reconstruct(), tol,
          ca.name() + " reconstruction");
      // Reloaded factors themselves must rebuild the truncated matrix.
      if (!cb.storedAsDense()) {
        double[][] fromFactors = CompressedMatrix.reconstructFromFactors(
            cb.rows(), cb.cols(), cb.rank(),
            cb.factorU(), cb.factorS(), cb.factorVt());
        assertMatrixClose(h, fromFactors, ca.reconstruct(), tol,
            ca.name() + " loaded factors rebuild");
      }
    }
  }

  interface IoRunnable {
    void run() throws IOException;
  }

  static void rethrowIo(IoRunnable body) {
    try {
      body.run();
    } catch (IOException e) {
      throw new CompressionException("io failure: " + e.getMessage(), e);
    }
  }
}
