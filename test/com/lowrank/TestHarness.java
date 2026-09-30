package com.lowrank;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/** Minimal JUnit-free test harness used by verify.sh. */
final class TestHarness {
  private final List<String> failures = new ArrayList<>();
  private int checks;

  void check(boolean condition, String message) {
    checks++;
    if (!condition) {
      failures.add(message);
    }
  }

  void checkEq(long actual, long expected, String message) {
    check(actual == expected, message + " (expected " + expected + ", got " + actual + ")");
  }

  void checkClose(double actual, double expected, double tol, String message) {
    double gap = Math.abs(actual - expected);
    checks++;
    if (!(gap <= tol)) {
      failures.add(message + " (expected " + expected + ", got " + actual
          + ", tol " + tol + ")");
    }
  }

  void expectThrows(Class<? extends Throwable> type, Runnable body, String message) {
    checks++;
    try {
      body.run();
    } catch (RuntimeException e) {
      if (type.isInstance(e)) {
        return;
      }
      failures.add(message + " (wrong exception " + e.getClass().getSimpleName()
          + ": " + e.getMessage() + ")");
      return;
    }
    failures.add(message + " (no exception thrown)");
  }

  interface ThrowingRunnable {
    void run() throws Exception;
  }

  void expectThrowsAny(Class<? extends Throwable> type, ThrowingRunnable body,
                       String message) {
    checks++;
    try {
      body.run();
    } catch (Throwable e) {
      if (type.isInstance(e)) {
        return;
      }
      failures.add(message + " (wrong exception " + e.getClass().getSimpleName()
          + ": " + e.getMessage() + ")");
      return;
    }
    failures.add(message + " (no exception thrown)");
  }

  int failureCount() {
    return failures.size();
  }

  int checkCount() {
    return checks;
  }

  List<String> failures() {
    return failures;
  }

  static void run(String name, Consumer<TestHarness> body) {
    TestHarness h = new TestHarness();
    try {
      body.accept(h);
    } catch (RuntimeException e) {
      h.check(false, name + " threw " + e.getClass().getSimpleName() + ": " + e.getMessage());
    }
    if (h.failureCount() == 0) {
      System.out.printf("PASS %-44s (%d assertions)%n", name, h.checkCount());
    } else {
      System.out.printf("FAIL %-44s%n", name);
      for (String f : h.failures()) {
        System.out.println("       - " + f);
      }
      System.exit(1);
    }
  }
}
