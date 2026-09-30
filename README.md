# Non-Uniform Low-Rank Matrix Compression Engine

A small, dependency-light Java engine for Linux/x86_64 CPUs (JDK 8+ source
level; developed and verified on OpenJDK 17). It compresses a batch of named
small matrices (every dimension between 1 and 12) with one shared spectral
threshold while each matrix normalizes by its own largest singular value, so
different spectra keep different ranks.

## Run it

```bash
bash ./verify.sh
```

The script compiles every source and test file with `javac`, runs a non-zero
suite of automated tests (17 test groups, hundreds of assertions), and then
executes a small end-to-end demo. It needs no network and no external service.

## The one matrix library and offline preparation

Exactly one mature local matrix library is used: **EJML 0.43.1**
(Apache-2.0, pure Java), and only for the thin SVD (`SimpleMatrix.svd(true)`).
Everything else (normalization, truncation, storage, multiplication) is plain
Java.

The required jars are vendored under `lib/` (~880 KB total):

- `lib/ejml-core-0.43.1.jar`
- `lib/ejml-ddense-0.43.1.jar`
- `lib/ejml-simple-0.43.1.jar`

They are already present, so normal builds are offline. To (re)prepare them on
a machine with internet, run `bash lib/fetch_jars.sh`; it downloads the three
jars from Maven Central into `lib/`. For an air-gapped machine, copy that
`lib/` directory from any prepared checkout. Note `org.ejml:ejml-all` is an
empty aggregator POM artifact; the three module jars above are what actually
contain the classes.

## Model and semantics

For matrix A (m x n) a thin SVD is A = U diag(s) Vt with k = min(m,n)
directions. Normalization is per matrix: level_i = s_i / s_0.

- A **zero** singular value (s_i <= 1e-12 * s_0, or A = 0) is always deleted.
  No division by zero is ever performed; an all-zero matrix always has rank 0
  and its k directions count as deleted.
- A nonzero direction is retained iff `level_i >= threshold`; equality is
  retained (`0.5` at threshold `0.5` survives).
- Equal normalized values are clustered (tolerance 1e-9) and move as one unit:
  the engine never splits equal spectral values to use up a deletion budget.
- Both the threshold and the target deletion ratio are validated to be finite
  and in `[0,1]`.

**Batch denominator.** The deletion ratio is deleted directions divided by the
total number of thin-spectrum directions across the batch, zero directions
included.

**Discrete reachable cuts.** Because equal values stay together, the batch has
only a discrete set of reachable ratios. In target mode the engine selects the
reachable cut whose realized ratio is closest to the target; on an exact tie it
keeps more directions (deletes fewer). The report contains the requested
target, the actual ratio, and the signed deviation `actual - target` (plus its
absolute value), and a canonical threshold in `[0,1]` that reproduces the cut
through the threshold API.

**Spectral result vs. real storage gain.** These are reported separately:

- the spectral truncation always yields factors U (m*r), s (r), Vt (r*n),
  i.e. `r*(m+n+1)` doubles, versus the original `m*n`;
- if the factors do not use strictly fewer doubles (e.g. full-rank small
  matrices), the persistent snapshot stores the dense truncated matrix instead;
- each per-matrix report lists retained rank, relative Frobenius
  reconstruction error, factor element count, original element count, the
  chosen storage form, and whether storage genuinely saves space.

A rank-0 result stores zero data doubles; it reconstructs the zero matrix.

## Validation and safety

Rejected explicitly with `CompressionException`: empty batch, empty matrix,
ragged matrix, dimension > 12, empty/duplicate names, non-finite entries
(NaN/Infinity), and out-of-range threshold or target.

## Persistence and multiplication

`BatchResult.save(Path)` / `BatchResult.load(Path)` use a binary container
(magic `NLRM`, version, CRC32 checksum, payload). Saves are atomic: bytes go
to a temp file, are flushed, then moved over the destination; a failed or
corrupt batch never replaces an intact existing file, and a corrupt/truncated
snapshot is refused before any result object is produced. Tests assert that
reconstructions and matrix products are identical before and after reload
(reloaded factors themselves rebuild the truncated matrix).

`BatchResult.multiply("L","R")` multiplies the truncated reconstructions of two
batch matrices and rejects inner-dimension mismatches.

## Layout

- `src/com/lowrank/` - engine, SVD wrapper, result model, binary store, demo
- `test/com/lowrank/` - dependency-free test harness and behavior tests
- `lib/` - vendored EJML jars and `fetch_jars.sh`
- `verify.sh` - one-command compile + test + demo entry point

The test suite specifically covers: spectra producing different ranks,
discrete budgets with tie-breaking toward fewer deletions, equality retention,
no splitting of equal values, zero matrices and shared denominators,
reconstruction errors, dense fallback when factors do not pay off, save/load
round-trips, corrupt-file rejection, failed-save atomicity, and identical
multiplication across reload.
