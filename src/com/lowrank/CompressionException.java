package com.lowrank;

/**
 * Raised for invalid inputs, corrupt batch files, or dimension mismatches.
 */
public class CompressionException extends RuntimeException {
  public CompressionException(String message) {
    super(message);
  }

  public CompressionException(String message, Throwable cause) {
    super(message, cause);
  }
}
