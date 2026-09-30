package lowrank;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class ResultStore {
    private static final byte[] MAGIC = {'N', 'R', 'C', 'B'};
    private static final int VERSION = 1;

    private ResultStore() {
    }

    static void save(Path file, CompressionResult result) {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(result, "result");
        try {
            byte[] payload = encode(result);
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] checksum = digest.digest(payload);
            Path parent = file.toAbsolutePath().getParent();
            Path temporary = Files.createTempFile(parent, ".nrcb-", ".tmp");
            try {
                try (DataOutputStream output = new DataOutputStream(Files.newOutputStream(temporary))) {
                    output.write(MAGIC);
                    output.writeInt(VERSION);
                    output.write(checksum);
                    output.write(payload);
                }
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } finally {
                Files.deleteIfExists(temporary);
            }
        } catch (IOException | NoSuchAlgorithmException | RuntimeException exception) {
            throw new IllegalArgumentException("failed to save complete result: " + exception.getMessage(), exception);
        }
    }

    static CompressionResult load(Path file) {
        Objects.requireNonNull(file, "file");
        try {
            byte[] stored = Files.readAllBytes(file);
            if (stored.length < MAGIC.length + Integer.BYTES + 32) {
                throw new IOException("file is truncated");
            }
            for (int i = 0; i < MAGIC.length; i++) {
                if (stored[i] != MAGIC[i]) {
                    throw new IOException("unrecognized file signature");
                }
            }
            int version = readInt(stored, MAGIC.length);
            if (version != VERSION) {
                throw new IOException("unsupported file version: " + version);
            }
            byte[] storedChecksum = new byte[32];
            System.arraycopy(stored, 8, storedChecksum, 0, 32);
            byte[] payload = new byte[stored.length - 40];
            System.arraycopy(stored, 40, payload, 0, payload.length);
            byte[] actualChecksum = MessageDigest.getInstance("SHA-256").digest(payload);
            if (!MessageDigest.isEqual(storedChecksum, actualChecksum)) {
                throw new IOException("checksum mismatch");
            }
            CompressionResult result = decode(payload);
            validateLoaded(result);
            return result;
        } catch (IOException | NoSuchAlgorithmException | RuntimeException exception) {
            throw new IllegalArgumentException("failed to load result: " + exception.getMessage(), exception);
        }
    }

    private static byte[] encode(CompressionResult result) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream output = new DataOutputStream(bytes)) {
            output.writeByte(result.mode().ordinal());
            output.writeDouble(result.requestedThreshold());
            output.writeDouble(result.targetDeletedFraction());
            output.writeDouble(result.effectiveThreshold());
            output.writeInt(result.totalThinDirections());
            output.writeInt(result.deletedDirections());
            output.writeInt(result.matrices().size());
            for (CompressedMatrix matrix : result.matrices().values()) {
                writeString(output, matrix.name());
                output.writeInt(matrix.rows());
                output.writeInt(matrix.columns());
                output.writeInt(matrix.retainedRank());
                output.writeInt(matrix.zeroDirections());
                output.writeInt(matrix.deletedPositiveDirections());
                output.writeBoolean(matrix.factorStored());
                output.writeDouble(matrix.absoluteReconstructionError());
                output.writeDouble(matrix.relativeReconstructionError());
                if (matrix.factorStored()) {
                    writeDoubles(output, matrix.leftFactor());
                    writeDoubles(output, matrix.rightFactor());
                } else {
                    writeDoubles(output, matrix.denseMatrix().toRowMajor());
                }
            }
        }
        return bytes.toByteArray();
    }

    private static CompressionResult decode(byte[] payload) throws IOException {
        try (DataInputStream input = new DataInputStream(new ByteArrayInputStream(payload))) {
            int modeOrdinal = input.readByte();
            if (modeOrdinal < 0 || modeOrdinal >= CompressionMode.values().length) {
                throw new IOException("invalid compression mode");
            }
            CompressionMode mode = CompressionMode.values()[modeOrdinal];
            double requestedThreshold = input.readDouble();
            double target = input.readDouble();
            double effectiveThreshold = input.readDouble();
            int totalDirections = input.readInt();
            int deletedDirections = input.readInt();
            int count = input.readInt();
            if (count <= 0 || totalDirections <= 0 || deletedDirections < 0 || deletedDirections > totalDirections) {
                throw new IOException("invalid batch totals");
            }
            List<CompressedMatrix> matrices = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                String name = readString(input);
                int rows = input.readInt();
                int columns = input.readInt();
                int rank = input.readInt();
                int zeros = input.readInt();
                int deletedPositive = input.readInt();
                boolean factorStored = input.readBoolean();
                double absoluteError = input.readDouble();
                double relativeError = input.readDouble();
                validateShape(name, rows, columns, rank, zeros, deletedPositive, factorStored);
                if (absoluteError < 0.0 || relativeError < 0.0 || relativeError > 1.0) {
                    throw new IOException("invalid reconstruction errors for " + name);
                }
                double[] left = factorStored ? readDoubles(input) : new double[0];
                double[] right = factorStored ? readDoubles(input) : new double[0];
                double[] denseValues = factorStored ? new double[0] : readDoubles(input);
                if (factorStored) {
                    if (left.length != rows * rank || right.length != columns * rank) {
                        throw new IOException("invalid factor array length for " + name);
                    }
                } else if (denseValues.length != rows * columns) {
                    throw new IOException("invalid dense array length for " + name);
                }
                Matrix dense = factorStored ? null : new Matrix(rows, columns, denseValues);
                matrices.add(new CompressedMatrix(name, rows, columns, rank, zeros, deletedPositive,
                        factorStored, left, right, dense, absoluteError, relativeError));
            }
            int loadedDirections = matrices.stream().mapToInt(CompressedMatrix::thinDirections).sum();
            int loadedDeleted = matrices.stream().mapToInt(CompressedMatrix::deletedDirections).sum();
            if (loadedDirections != totalDirections || loadedDeleted != deletedDirections) {
                throw new IOException("per-matrix directions do not match batch totals");
            }
            return new CompressionResult(mode, requestedThreshold, target, effectiveThreshold,
                    totalDirections, deletedDirections, matrices);
        }
    }

    private static void validateLoaded(CompressionResult result) {
        if (!Double.isFinite(result.effectiveThreshold()) || result.effectiveThreshold() < 0.0
                || result.effectiveThreshold() > 1.0) {
            throw new IllegalStateException("invalid effective threshold");
        }
        if (result.mode() == CompressionMode.THRESHOLD
                && (!Double.isFinite(result.requestedThreshold())
                || result.requestedThreshold() != result.effectiveThreshold())) {
            throw new IllegalStateException("invalid requested threshold");
        }
        if (result.mode() == CompressionMode.TARGET_DELETION
                && (!Double.isFinite(result.targetDeletedFraction())
                || result.targetDeletedFraction() < 0.0
                || result.targetDeletedFraction() > 1.0
                || !Double.isFinite(result.deviation()))) {
            throw new IllegalStateException("invalid target fraction");
        }
        double computedFraction = (double) result.deletedDirections() / result.totalThinDirections();
        if (computedFraction != result.actualDeletedFraction()) {
            throw new IllegalStateException("invalid actual deleted fraction");
        }
    }

    private static void validateShape(String name, int rows, int columns, int rank,
                                      int zeros, int deletedPositive, boolean factorStored)
            throws IOException {
        if (name == null || name.isBlank() || rows < 1 || columns < 1 || rows > 12 || columns > 12) {
            throw new IOException("invalid matrix descriptor");
        }
        int thin = Math.min(rows, columns);
        if (rank < 0 || zeros < 0 || deletedPositive < 0
                || rank + zeros + deletedPositive != thin) {
            throw new IOException("invalid ranks for " + name);
        }
        int factorElements = rank * (rows + columns);
        if (factorStored && factorElements >= rows * columns) {
            throw new IOException("factor representation has no storage gain for " + name);
        }
    }

    private static void writeString(DataOutputStream output, String value) throws IOException {
        byte[] encoded = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        output.writeInt(encoded.length);
        output.write(encoded);
    }

    private static String readString(DataInputStream input) throws IOException {
        int length = input.readInt();
        if (length <= 0 || length > 256) {
            throw new IOException("invalid name length");
        }
        byte[] encoded = new byte[length];
        input.readFully(encoded);
        String value = new String(encoded, java.nio.charset.StandardCharsets.UTF_8);
        if (value.isBlank()) {
            throw new IOException("blank matrix name");
        }
        return value;
    }

    private static void writeDoubles(DataOutputStream output, double[] values) throws IOException {
        output.writeInt(values.length);
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IOException("non-finite stored value");
            }
            output.writeDouble(value);
        }
    }

    private static double[] readDoubles(DataInputStream input) throws IOException {
        int length = input.readInt();
        if (length < 0) {
            throw new IOException("invalid value array length");
        }
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            double value = input.readDouble();
            if (!Double.isFinite(value)) {
                throw new IOException("non-finite stored value");
            }
            values[i] = value;
        }
        return values;
    }

    private static int readInt(byte[] bytes, int offset) {
        return ((bytes[offset] & 0xff) << 24)
                | ((bytes[offset + 1] & 0xff) << 16)
                | ((bytes[offset + 2] & 0xff) << 8)
                | (bytes[offset + 3] & 0xff);
    }
}
