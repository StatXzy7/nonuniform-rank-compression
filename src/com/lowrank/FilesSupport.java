package com.lowrank;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Tiny helper for the demo so Demo stays free of filesystem boilerplate. */
final class FilesSupport {
  private FilesSupport() {
  }

  static Path tempPath(String prefix, String suffix) throws IOException {
    Path p = Files.createTempFile(prefix, suffix);
    Files.deleteIfExists(p);
    return p;
  }
}
