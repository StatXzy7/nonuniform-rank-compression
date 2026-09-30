package com.lowrank;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable result of compressing one batch of matrices.
 *
 * <p>Carries the request mode, the requested threshold/target ratio, the
 * effective threshold actually applied, and the realized deletion ratio over
 * the total number of thin-spectrum directions in the batch (zero directions
 * included).
 */
public final class BatchResult {
  private final Mode mode;
  private final double requested;
  private final double effectiveThreshold;
  private final double actualDeletedRatio;
  private final double deviation;
  private final int totalDirections;
  private final int deletedDirections;
  private final Map<String, CompressedMatrix> matrices;

  enum Mode { THRESHOLD, TARGET_RATIO }

  BatchResult(Mode mode, double requested, double effectiveThreshold,
              int totalDirections, int deletedDirections,
              LinkedHashMap<String, CompressedMatrix> matrices) {
    this.mode = mode;
    this.requested = requested;
    this.effectiveThreshold = effectiveThreshold;
    this.totalDirections = totalDirections;
    this.deletedDirections = deletedDirections;
    this.actualDeletedRatio = totalDirections == 0
        ? 0.0 : (double) deletedDirections / totalDirections;
    this.deviation = mode == Mode.TARGET_RATIO
        ? this.actualDeletedRatio - requested : 0.0;
    this.matrices = Collections.unmodifiableMap(new LinkedHashMap<>(matrices));
  }

  /** "THRESHOLD" or "TARGET_RATIO". */
  public String modeName() {
    return mode == Mode.THRESHOLD ? "THRESHOLD" : "TARGET_RATIO";
  }

  boolean isTargetMode() {
    return mode == Mode.TARGET_RATIO;
  }

  /** Requested threshold (threshold mode) or target deletion ratio (target mode). */
  public double requestedValue() {
    return requested;
  }

  /** Canonical threshold that yields exactly this result. */
  public double effectiveThreshold() {
    return effectiveThreshold;
  }

  public int totalDirections() {
    return totalDirections;
  }

  public int deletedDirections() {
    return deletedDirections;
  }

  public int retainedDirections() {
    return totalDirections - deletedDirections;
  }

  /** Deleted thin-spectrum directions / all thin-spectrum directions. */
  public double actualDeletedRatio() {
    return actualDeletedRatio;
  }

  /** Signed gap (actual - target); defined in target mode, 0.0 otherwise. */
  public double deviation() {
    return deviation;
  }

  /** Absolute gap between actual and target; 0.0 in threshold mode. */
  public double absoluteDeviation() {
    return Math.abs(deviation);
  }

  public List<CompressedMatrix> matrices() {
    return new ArrayList<>(matrices.values());
  }

  public CompressedMatrix matrix(String name) {
    CompressedMatrix m = matrices.get(name);
    if (m == null) {
      throw new CompressionException("no matrix named '" + name + "' in batch");
    }
    return m;
  }

  /**
   * Multiplies the truncated reconstructions of two matrices in the batch.
   *
   * @throws CompressionException if the inner dimensions do not match.
   */
  public double[][] multiply(String leftName, String rightName) {
    CompressedMatrix left = matrix(leftName);
    CompressedMatrix right = matrix(rightName);
    if (left.cols() != right.rows()) {
      throw new CompressionException(
          "cannot multiply " + leftName + " (" + left.rows() + "x" + left.cols()
              + ") by " + rightName + " (" + right.rows() + "x" + right.cols() + ")");
    }
    return CompressedMatrix.multiply(left.reconstruct(), right.reconstruct());
  }

  /**
   * Atomically saves the batch: bytes are written and flushed to a temporary
   * file first, then atomically moved over the destination. Any failure leaves
   * a previously existing destination file untouched.
   */
  public void save(Path destination) throws IOException {
    if (destination == null) {
      throw new CompressionException("destination path must not be null");
    }
    Path parent = destination.toAbsolutePath().getParent();
    Path tmp = Files.createTempFile(parent, destination.getFileName() + ".", ".tmp");
    boolean moved = false;
    try {
      try (OutputStream out = Files.newOutputStream(tmp)) {
        ResultStore.write(this, out);
        out.flush();
      }
      try {
        Files.move(tmp, destination, StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE);
      } catch (IOException atomicFailed) {
        Files.move(tmp, destination, StandardCopyOption.REPLACE_EXISTING);
      }
      moved = true;
    } finally {
      if (!moved) {
        try {
          Files.deleteIfExists(tmp);
        } catch (IOException ignored) {
          // best effort; the destination is never touched on failure
        }
      }
    }
  }

  public static BatchResult load(Path source) throws IOException {
    try (java.io.InputStream in = Files.newInputStream(source)) {
      return ResultStore.read(in);
    }
  }
}
