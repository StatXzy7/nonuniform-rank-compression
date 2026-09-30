package com.lowrank;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Test-only seam: mirrors BatchResult.save but with an injectable stream. */
final class ResultStoreTestSupport {
  private ResultStoreTestSupport() {
  }

  static void saveFailing(BatchResult result, Path destination, OutputStream failing)
      throws IOException {
    Path parent = destination.toAbsolutePath().getParent();
    Path tmp = Files.createTempFile(parent, destination.getFileName() + ".", ".tmp");
    boolean moved = false;
    try {
      try (OutputStream out = failing) {
        ResultStore.write(result, out);
        out.flush();
      }
      Files.move(tmp, destination, StandardCopyOption.REPLACE_EXISTING,
          StandardCopyOption.ATOMIC_MOVE);
      moved = true;
    } finally {
      if (!moved) {
        Files.deleteIfExists(tmp);
      }
    }
  }
}
