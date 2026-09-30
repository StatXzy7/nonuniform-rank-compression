package lowrank;

import java.util.Objects;

public record NamedMatrix(String name, Matrix matrix) {
    public NamedMatrix {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(matrix, "matrix");
        if (name.isBlank()) {
            throw new IllegalArgumentException("matrix name must not be blank");
        }
    }
}
