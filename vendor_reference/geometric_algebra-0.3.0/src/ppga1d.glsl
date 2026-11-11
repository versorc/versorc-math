struct DualNumber {
    // 1, e01
    vec2 g0;
};

float scalar_zero() {
    return 0.0;
}

float scalar_one() {
    return 1.0;
}

float scalar_neg(float self) {
    return self * -1.0;
}

float scalar_automorphism(float self) {
    return self;
}

float scalar_reversal(float self) {
    return self;
}

float scalar_conjugation(float self) {
    return self;
}

float scalar_scalar_add(float self, float other) {
    return self + other;
}

float scalar_scalar_sub(float self, float other) {
    return self - other;
}

float scalar_scalar_mul(float self, float other) {
    return self * other;
}

float scalar_scalar_div(float self, float other) {
    return self * 1.0 / other * 1.0;
}

float scalar_scalar_geometric_product(float self, float other) {
    return self * other;
}

float scalar_scalar_outer_product(float self, float other) {
    return self * other;
}

float scalar_scalar_inner_product(float self, float other) {
    return self * other;
}

float scalar_scalar_left_contraction(float self, float other) {
    return self * other;
}

float scalar_scalar_right_contraction(float self, float other) {
    return self * other;
}

float scalar_scalar_scalar_product(float self, float other) {
    return self * other;
}

DualNumber scalar_dual_number_add(float self, DualNumber other) {
    return DualNumber(vec2(self) * vec2(1.0, 0.0) + other.g0);
}

DualNumber scalar_dual_number_sub(float self, DualNumber other) {
    return DualNumber(vec2(self) * vec2(1.0, 0.0) - other.g0);
}

DualNumber scalar_dual_number_geometric_product(float self, DualNumber other) {
    return DualNumber(vec2(self) * other.g0);
}

float scalar_dual_number_regressive_product(float self, DualNumber other) {
    return self * other.g0.y;
}

DualNumber scalar_dual_number_outer_product(float self, DualNumber other) {
    return DualNumber(vec2(self) * other.g0);
}

DualNumber scalar_dual_number_inner_product(float self, DualNumber other) {
    return DualNumber(vec2(self) * other.g0);
}

DualNumber scalar_dual_number_left_contraction(float self, DualNumber other) {
    return DualNumber(vec2(self) * other.g0);
}

float scalar_dual_number_right_contraction(float self, DualNumber other) {
    return self * other.g0.x;
}

float scalar_dual_number_scalar_product(float self, DualNumber other) {
    return self * other.g0.x;
}

float scalar_squared_magnitude(float self) {
    return scalar_scalar_scalar_product(self, scalar_reversal(self));
}

float scalar_magnitude(float self) {
    return sqrt(scalar_squared_magnitude(self));
}

float scalar_signum(float self) {
    return scalar_scalar_geometric_product(self, 1.0 / scalar_magnitude(self));
}

float scalar_inverse(float self) {
    return scalar_scalar_geometric_product(scalar_reversal(self), 1.0 / scalar_squared_magnitude(self));
}

DualNumber dual_number_zero() {
    return DualNumber(vec2(0.0));
}

DualNumber dual_number_one() {
    return DualNumber(vec2(1.0, 0.0));
}

DualNumber dual_number_neg(DualNumber self) {
    return DualNumber(self.g0 * vec2(-1.0));
}

DualNumber dual_number_automorphism(DualNumber self) {
    return DualNumber(self.g0);
}

DualNumber dual_number_reversal(DualNumber self) {
    return DualNumber(self.g0 * vec2(1.0, -1.0));
}

DualNumber dual_number_conjugation(DualNumber self) {
    return DualNumber(self.g0 * vec2(1.0, -1.0));
}

DualNumber dual_number_dual(DualNumber self) {
    return DualNumber(self.g0.yx);
}

float dual_number_scalar_into(DualNumber self) {
    return self.g0.x;
}

DualNumber dual_number_scalar_add(DualNumber self, float other) {
    return DualNumber(self.g0 + vec2(other) * vec2(1.0, 0.0));
}

DualNumber dual_number_scalar_sub(DualNumber self, float other) {
    return DualNumber(self.g0 - vec2(other) * vec2(1.0, 0.0));
}

DualNumber dual_number_scalar_geometric_product(DualNumber self, float other) {
    return DualNumber(self.g0 * vec2(other));
}

float dual_number_scalar_regressive_product(DualNumber self, float other) {
    return self.g0.y * other;
}

DualNumber dual_number_scalar_outer_product(DualNumber self, float other) {
    return DualNumber(self.g0 * vec2(other));
}

DualNumber dual_number_scalar_inner_product(DualNumber self, float other) {
    return DualNumber(self.g0 * vec2(other));
}

float dual_number_scalar_left_contraction(DualNumber self, float other) {
    return self.g0.x * other;
}

DualNumber dual_number_scalar_right_contraction(DualNumber self, float other) {
    return DualNumber(self.g0 * vec2(other));
}

float dual_number_scalar_scalar_product(DualNumber self, float other) {
    return self.g0.x * other;
}

DualNumber dual_number_dual_number_add(DualNumber self, DualNumber other) {
    return DualNumber(self.g0 + other.g0);
}

DualNumber dual_number_dual_number_sub(DualNumber self, DualNumber other) {
    return DualNumber(self.g0 - other.g0);
}

DualNumber dual_number_dual_number_mul(DualNumber self, DualNumber other) {
    return DualNumber(self.g0 * other.g0);
}

DualNumber dual_number_dual_number_div(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.x, self.g0.y) * vec2(1.0, 1.0) / vec2(other.g0.x, other.g0.y) * vec2(1.0, 1.0));
}

DualNumber dual_number_dual_number_geometric_product(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.x) * other.g0 + self.g0 * vec2(other.g0.x) * vec2(0.0, 1.0));
}

DualNumber dual_number_dual_number_regressive_product(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.y) * other.g0 + vec2(self.g0.x) * other.g0.yx * vec2(1.0, 0.0));
}

DualNumber dual_number_dual_number_outer_product(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.x) * other.g0 + self.g0 * vec2(other.g0.x) * vec2(0.0, 1.0));
}

DualNumber dual_number_dual_number_inner_product(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.x) * other.g0 + self.g0 * vec2(other.g0.x) * vec2(0.0, 1.0));
}

DualNumber dual_number_dual_number_left_contraction(DualNumber self, DualNumber other) {
    return DualNumber(vec2(self.g0.x) * other.g0);
}

DualNumber dual_number_dual_number_right_contraction(DualNumber self, DualNumber other) {
    return DualNumber(self.g0 * vec2(other.g0.x));
}

float dual_number_dual_number_scalar_product(DualNumber self, DualNumber other) {
    return self.g0.x * other.g0.x;
}

float dual_number_squared_magnitude(DualNumber self) {
    return dual_number_dual_number_scalar_product(self, dual_number_reversal(self));
}

float dual_number_magnitude(DualNumber self) {
    return sqrt(dual_number_squared_magnitude(self));
}

DualNumber dual_number_scalar_mul(DualNumber self, float other) {
    return dual_number_scalar_geometric_product(self, other);
}

DualNumber dual_number_signum(DualNumber self) {
    return dual_number_scalar_geometric_product(self, 1.0 / dual_number_magnitude(self));
}

DualNumber dual_number_inverse(DualNumber self) {
    return dual_number_scalar_geometric_product(dual_number_reversal(self), 1.0 / dual_number_squared_magnitude(self));
}

DualNumber dual_number_powi(DualNumber self, int exponent) {
    if(exponent == 0) {
        return dual_number_one();
    }
    DualNumber x = (exponent < 0) ? dual_number_inverse(self) : self;
    DualNumber y = dual_number_one();
    int n = abs(exponent);
    while(1 < n) {
        if((n & 1) == 1) {
            y = dual_number_dual_number_geometric_product(x, y);
        }
        x = dual_number_dual_number_geometric_product(x, x);
        n = n >> 1;
    }
    return dual_number_dual_number_geometric_product(x, y);
}

DualNumber dual_number_dual_number_geometric_quotient(DualNumber self, DualNumber other) {
    return dual_number_dual_number_geometric_product(self, dual_number_inverse(other));
}

DualNumber dual_number_dual_number_transformation(DualNumber self, DualNumber other) {
    return dual_number_dual_number_geometric_product(dual_number_dual_number_geometric_product(self, other), dual_number_reversal(self));
}

DualNumber dual_number_scalar_geometric_quotient(DualNumber self, float other) {
    return dual_number_scalar_geometric_product(self, scalar_inverse(other));
}

float dual_number_scalar_transformation(DualNumber self, float other) {
    return dual_number_scalar_into(dual_number_dual_number_geometric_product(dual_number_scalar_geometric_product(self, other), dual_number_reversal(self)));
}

DualNumber scalar_dual_number_geometric_quotient(float self, DualNumber other) {
    return scalar_dual_number_geometric_product(self, dual_number_inverse(other));
}

DualNumber scalar_dual_number_transformation(float self, DualNumber other) {
    return dual_number_scalar_geometric_product(scalar_dual_number_geometric_product(self, other), scalar_reversal(self));
}

float scalar_powi(float self, int exponent) {
    if(exponent == 0) {
        return scalar_one();
    }
    float x = (exponent < 0) ? scalar_inverse(self) : self;
    float y = scalar_one();
    int n = abs(exponent);
    while(1 < n) {
        if((n & 1) == 1) {
            y = scalar_scalar_geometric_product(x, y);
        }
        x = scalar_scalar_geometric_product(x, x);
        n = n >> 1;
    }
    return scalar_scalar_geometric_product(x, y);
}

float scalar_scalar_geometric_quotient(float self, float other) {
    return scalar_scalar_geometric_product(self, scalar_inverse(other));
}

float scalar_scalar_transformation(float self, float other) {
    return scalar_scalar_geometric_product(scalar_scalar_geometric_product(self, other), scalar_reversal(self));
}

