package com.lowrank;

import java.util.Arrays;

/**
 * An immutable dense matrix with a non-empty name.
 *
 * <p>Every dimension must be between 1 and 12 and every entry must be finite.
 */
public final class NamedMatrix {
  /** Largest permitted dimension for any axis. */
  public static final int MAX_DIM = 12;

  private final String name;
  private final double[][] data;

  public NamedMatrix(String name, double[][] data) {
    if (name == null || name.isEmpty()) {
      throw new CompressionException("matrix name must be a non-empty string");
    }
    if (data == null || data.length == 0) {
      throw new CompressionException("matrix '" + name + "' must have at least one row");
    }
    int rows = data.length;
    if (rows > MAX_DIM) {
      throw new CompressionException(
          "matrix '" + name + "' has " + rows + " rows, maximum is " + MAX_DIM);
    }
    Integer cols = null;
    for (double[] row : data) {
      if (row == null || row.length == 0) {
        throw new CompressionException("matrix '" + name + "' must have at least one column");
      }
      if (cols == null) {
        cols = row.length;
      } else if (row.length != cols) {
          throw new CompressionException("matrix '" + name + "' has ragged rows");
      }
      if (cols > MAX_DIM) {
        throw new CompressionException(
            "matrix '" + name + "' has " + cols + " columns, maximum is " + MAX_DIM);
      }
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (!Double.isFinite(data[i][j])) {
          throw new CompressionException(
              "matrix '" + name + "' contains a non-finite value at [" + i + "," + j + "]");
        }
      }
    }
    this.name = name;
    this.data = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
      this.data[i] = Arrays.copyOf(data[i], cols);
    }
  }

  public String name() {
    return name;
  }

  public int rows() {
    return data.length;
  }

  public int cols() {
    return data[0].length;
  }

  /** Defensive copy of the underlying entries. */
  public double[][] toArray() {
    double[][] out = new double[data.length][];
    for (int i = 0; i < data.length; i++) {
      out[i] = Arrays.copyOf(data[i], data[i].length);
    }
    return out;
  }

  double get(int row, int col) {
    return data[row][col];
  }
}
