package com.lowrank;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.zip.CRC32;

/**
 * Binary snapshot format for {@link BatchResult}.
 *
 * <p>Layout: magic "NLRM", format version (1), a 4-byte CRC32 of the payload,
 * and the big-endian payload described below. A truncated, truncated-by-
 * failure, or otherwise corrupt stream is rejected before any result object is
 * produced, so callers never observe a half-loaded batch.
 *
 * <pre>
 *   int   mode (0 = threshold, 1 = target ratio)
 *   double requested
 *   double effectiveThreshold
 *   int   totalDirections
 *   int   deletedDirections
 *   int   matrixCount
 *   per matrix:
 *     UTF name, int rows, int cols, int rank, double relError, byte dense
 *     dense == 1 : rows*cols doubles (truncated matrix)
 *     dense == 0 : rank singular values, then m*rank U doubles, then rank*n Vt doubles
 * </pre>
 */
final class ResultStore {
  private static final int MAGIC = 0x4E4C524D; // "NLRM"
  private static final int VERSION = 1;

  private ResultStore() {
  }

  static void write(BatchResult result, OutputStream out) throws IOException {
    ByteArrayOutputStream payload = new ByteArrayOutputStream();
    try (DataOutputStream data = new DataOutputStream(payload)) {
      data.writeInt(result.isTargetMode() ? 1 : 0);
      data.writeDouble(result.requestedValue());
      data.writeDouble(result.effectiveThreshold());
      data.writeInt(result.totalDirections());
      data.writeInt(result.deletedDirections());
      java.util.List<CompressedMatrix> matrices = result.matrices();
      data.writeInt(matrices.size());
      for (CompressedMatrix cm : matrices) {
        data.writeUTF(cm.name());
        data.writeInt(cm.rows());
        data.writeInt(cm.cols());
        data.writeInt(cm.rank());
        data.writeDouble(cm.relativeFroError());
        data.writeBoolean(cm.storedAsDense());
        if (cm.storedAsDense()) {
          writeMatrix(data, cm.truncatedArray(), cm.rows(), cm.cols());
        } else {
          double[] s = cm.factorS();
          for (double v : s) {
            data.writeDouble(v);
          }
          writeMatrix(data, cm.factorU(), cm.rows(), cm.rank());
          writeMatrix(data, cm.factorVt(), cm.rank(), cm.cols());
        }
      }
    }
    byte[] body = payload.toByteArray();
    CRC32 crc = new CRC32();
    crc.update(body);

    DataOutputStream header = new DataOutputStream(out);
    header.writeInt(MAGIC);
    header.writeInt(VERSION);
    header.writeInt((int) crc.getValue());
    header.writeInt(body.length);
    out.write(body);
    out.flush();
  }

  private static void writeMatrix(DataOutputStream data, double[][] a,
                                  int rows, int cols) throws IOException {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data.writeDouble(a[i][j]);
      }
    }
  }

  static BatchResult read(InputStream in) throws IOException {
    DataInputStream data = new DataInputStream(in);
    int magic;
    try {
      magic = data.readInt();
    } catch (IOException eof) {
      throw new CompressionException("corrupt snapshot: stream is empty", eof);
    }
    if (magic != MAGIC) {
      throw new CompressionException("corrupt snapshot: bad magic number");
    }
    int version = data.readInt();
    if (version != VERSION) {
      throw new CompressionException("unsupported snapshot version: " + version);
    }
    int expectedCrc = data.readInt();
    int length = data.readInt();
    if (length < 0) {
      throw new CompressionException("corrupt snapshot: negative payload length");
    }
    byte[] body = new byte[length];
    data.readFully(body);
    CRC32 crc = new CRC32();
    crc.update(body);
    if ((int) crc.getValue() != expectedCrc) {
      throw new CompressionException("corrupt snapshot: checksum mismatch");
    }
    try {
      return parse(body);
    } catch (CompressionException e) {
      throw e;
    } catch (RuntimeException e) {
      throw new CompressionException("corrupt snapshot: " + e.getMessage(), e);
    }
  }

  private static BatchResult parse(byte[] body) throws IOException {
    try (DataInputStream d = new DataInputStream(new ByteArrayInputStream(body))) {
      int modeCode = d.readInt();
      if (modeCode < 0 || modeCode > 1) {
        throw new CompressionException("bad mode code " + modeCode);
      }
      BatchResult.Mode mode =
          modeCode == 1 ? BatchResult.Mode.TARGET_RATIO : BatchResult.Mode.THRESHOLD;
      double requested = d.readDouble();
      double effective = d.readDouble();
      int totalDirs = d.readInt();
      int deletedDirs = d.readInt();
      if (!Double.isFinite(requested) || !Double.isFinite(effective)) {
        throw new CompressionException("non-finite control values");
      }
      int total = totalDirs;
      int deleted = deletedDirs;
      if (total < 0 || deleted < 0 || deleted > total) {
        throw new CompressionException("inconsistent direction counts");
      }
      int count = d.readInt();
      if (count <= 0) {
        throw new CompressionException("snapshot must contain matrices");
      }
      LinkedHashMap<String, CompressedMatrix> matrices = new LinkedHashMap<>();
      int parsedDirs = 0;
      int parsedDeleted = 0;
      for (int q = 0; q < count; q++) {
        String name = d.readUTF();
        if (name.isEmpty() || matrices.containsKey(name)) {
          throw new CompressionException("bad or duplicate name in snapshot");
        }
        int m = d.readInt();
        int n = d.readInt();
        int rank = d.readInt();
        double relError = d.readDouble();
        boolean dense = d.readBoolean();
        validateDims(name, m, n, rank, relError);
        int k = Math.min(m, n);
        parsedDirs += k;

        double[][] truncated;
        double[][] u;
        double[] s;
        double[][] vt;
        if (dense) {
          truncated = readDoubles(d, m, n);
          s = new double[rank];
          u = new double[m][rank];
          vt = new double[rank][n];
        } else {
          s = new double[rank];
          for (int r = 0; r < rank; r++) {
            s[r] = d.readDouble();
            if (!Double.isFinite(s[r]) || s[r] < 0.0) {
              throw new CompressionException("bad singular value in snapshot");
            }
          }
          u = readDoubles(d, m, rank);
          vt = readDoubles(d, rank, n);
          truncated = CompressedMatrix.reconstructFromFactors(m, n, rank, u, s, vt);
        }
        parsedDeleted += k - rank;
        CompressedMatrix cm = new CompressedMatrix(name, m, n, rank,
            truncated, u, s, vt, relError, dense);
        matrices.put(name, cm);
      }
      if (parsedDirs != total || parsedDeleted != deleted) {
        throw new CompressionException("direction counts do not match payload");
      }
      return new BatchResult(mode, requested, effective, total, deleted, matrices);
    }
  }

  private static void validateDims(String name, int m, int n, int rank,
                                   double relError) {
    if (m < 1 || n < 1 || m > NamedMatrix.MAX_DIM || n > NamedMatrix.MAX_DIM) {
      throw new CompressionException("bad dimensions for '" + name + "'");
    }
    if (rank < 0 || rank > Math.min(m, n)) {
      throw new CompressionException("bad rank for '" + name + "'");
    }
    if (!Double.isFinite(relError) || relError < 0.0 || relError > 1.0000001) {
      throw new CompressionException("bad reconstruction error for '" + name + "'");
    }
  }

  private static double[][] readDoubles(DataInputStream d, int rows, int cols)
      throws IOException {
    double[][] a = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        double v = d.readDouble();
        if (!Double.isFinite(v)) {
          throw new CompressionException("non-finite value in snapshot");
        }
        a[i][j] = v;
      }
    }
    return a;
  }
}
