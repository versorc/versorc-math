struct SplitComplexNumber {
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

SplitComplexNumber scalar_split_complex_number_add(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * vec2(1.0, 0.0) + other.g0);
}

SplitComplexNumber scalar_split_complex_number_sub(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * vec2(1.0, 0.0) - other.g0);
}

SplitComplexNumber scalar_split_complex_number_geometric_product(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * other.g0);
}

float scalar_split_complex_number_regressive_product(float self, SplitComplexNumber other) {
    return self * other.g0.y;
}

SplitComplexNumber scalar_split_complex_number_outer_product(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * other.g0);
}

SplitComplexNumber scalar_split_complex_number_inner_product(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * other.g0);
}

SplitComplexNumber scalar_split_complex_number_left_contraction(float self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self) * other.g0);
}

float scalar_split_complex_number_right_contraction(float self, SplitComplexNumber other) {
    return self * other.g0.x;
}

float scalar_split_complex_number_scalar_product(float self, SplitComplexNumber other) {
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

SplitComplexNumber split_complex_number_zero() {
    return SplitComplexNumber(vec2(0.0));
}

SplitComplexNumber split_complex_number_one() {
    return SplitComplexNumber(vec2(1.0, 0.0));
}

SplitComplexNumber split_complex_number_neg(SplitComplexNumber self) {
    return SplitComplexNumber(self.g0 * vec2(-1.0));
}

SplitComplexNumber split_complex_number_automorphism(SplitComplexNumber self) {
    return SplitComplexNumber(self.g0);
}

SplitComplexNumber split_complex_number_reversal(SplitComplexNumber self) {
    return SplitComplexNumber(self.g0 * vec2(1.0, -1.0));
}

SplitComplexNumber split_complex_number_conjugation(SplitComplexNumber self) {
    return SplitComplexNumber(self.g0 * vec2(1.0, -1.0));
}

SplitComplexNumber split_complex_number_dual(SplitComplexNumber self) {
    return SplitComplexNumber(self.g0.yx);
}

float split_complex_number_scalar_into(SplitComplexNumber self) {
    return self.g0.x;
}

SplitComplexNumber split_complex_number_scalar_add(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 + vec2(other) * vec2(1.0, 0.0));
}

SplitComplexNumber split_complex_number_scalar_sub(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 - vec2(other) * vec2(1.0, 0.0));
}

SplitComplexNumber split_complex_number_scalar_geometric_product(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 * vec2(other));
}

float split_complex_number_scalar_regressive_product(SplitComplexNumber self, float other) {
    return self.g0.y * other;
}

SplitComplexNumber split_complex_number_scalar_outer_product(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 * vec2(other));
}

SplitComplexNumber split_complex_number_scalar_inner_product(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 * vec2(other));
}

float split_complex_number_scalar_left_contraction(SplitComplexNumber self, float other) {
    return self.g0.x * other;
}

SplitComplexNumber split_complex_number_scalar_right_contraction(SplitComplexNumber self, float other) {
    return SplitComplexNumber(self.g0 * vec2(other));
}

float split_complex_number_scalar_scalar_product(SplitComplexNumber self, float other) {
    return self.g0.x * other;
}

SplitComplexNumber split_complex_number_split_complex_number_add(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(self.g0 + other.g0);
}

SplitComplexNumber split_complex_number_split_complex_number_sub(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(self.g0 - other.g0);
}

SplitComplexNumber split_complex_number_split_complex_number_mul(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(self.g0 * other.g0);
}

SplitComplexNumber split_complex_number_split_complex_number_div(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.x, self.g0.y) * vec2(1.0, 1.0) / vec2(other.g0.x, other.g0.y) * vec2(1.0, 1.0));
}

SplitComplexNumber split_complex_number_split_complex_number_geometric_product(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.x) * other.g0 + vec2(self.g0.y) * other.g0.yx);
}

SplitComplexNumber split_complex_number_split_complex_number_regressive_product(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.y) * other.g0 + vec2(self.g0.x) * other.g0.yx * vec2(1.0, 0.0));
}

SplitComplexNumber split_complex_number_split_complex_number_outer_product(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.x) * other.g0 + self.g0 * vec2(other.g0.x) * vec2(0.0, 1.0));
}

SplitComplexNumber split_complex_number_split_complex_number_inner_product(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.x) * other.g0 + vec2(self.g0.y) * other.g0.yx);
}

SplitComplexNumber split_complex_number_split_complex_number_left_contraction(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.x) * other.g0 + self.g0.yx * other.g0.yx * vec2(1.0, 0.0));
}

SplitComplexNumber split_complex_number_split_complex_number_right_contraction(SplitComplexNumber self, SplitComplexNumber other) {
    return SplitComplexNumber(vec2(self.g0.y) * other.g0.yx + vec2(self.g0.x) * vec2(other.g0.x) * vec2(1.0, 0.0));
}

float split_complex_number_split_complex_number_scalar_product(SplitComplexNumber self, SplitComplexNumber other) {
    return self.g0.x * other.g0.x + self.g0.y * other.g0.y;
}

float split_complex_number_squared_magnitude(SplitComplexNumber self) {
    return split_complex_number_split_complex_number_scalar_product(self, split_complex_number_reversal(self));
}

float split_complex_number_magnitude(SplitComplexNumber self) {
    return sqrt(split_complex_number_squared_magnitude(self));
}

SplitComplexNumber split_complex_number_scalar_mul(SplitComplexNumber self, float other) {
    return split_complex_number_scalar_geometric_product(self, other);
}

SplitComplexNumber split_complex_number_signum(SplitComplexNumber self) {
    return split_complex_number_scalar_geometric_product(self, 1.0 / split_complex_number_magnitude(self));
}

SplitComplexNumber split_complex_number_inverse(SplitComplexNumber self) {
    return split_complex_number_scalar_geometric_product(split_complex_number_reversal(self), 1.0 / split_complex_number_squared_magnitude(self));
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

SplitComplexNumber scalar_split_complex_number_geometric_quotient(float self, SplitComplexNumber other) {
    return scalar_split_complex_number_geometric_product(self, split_complex_number_inverse(other));
}

SplitComplexNumber scalar_split_complex_number_transformation(float self, SplitComplexNumber other) {
    return split_complex_number_scalar_geometric_product(scalar_split_complex_number_geometric_product(self, other), scalar_reversal(self));
}

SplitComplexNumber split_complex_number_scalar_geometric_quotient(SplitComplexNumber self, float other) {
    return split_complex_number_scalar_geometric_product(self, scalar_inverse(other));
}

float split_complex_number_scalar_transformation(SplitComplexNumber self, float other) {
    return split_complex_number_scalar_into(split_complex_number_split_complex_number_geometric_product(split_complex_number_scalar_geometric_product(self, other), split_complex_number_reversal(self)));
}

SplitComplexNumber split_complex_number_powi(SplitComplexNumber self, int exponent) {
    if(exponent == 0) {
        return split_complex_number_one();
    }
    SplitComplexNumber x = (exponent < 0) ? split_complex_number_inverse(self) : self;
    SplitComplexNumber y = split_complex_number_one();
    int n = abs(exponent);
    while(1 < n) {
        if((n & 1) == 1) {
            y = split_complex_number_split_complex_number_geometric_product(x, y);
        }
        x = split_complex_number_split_complex_number_geometric_product(x, x);
        n = n >> 1;
    }
    return split_complex_number_split_complex_number_geometric_product(x, y);
}

SplitComplexNumber split_complex_number_split_complex_number_geometric_quotient(SplitComplexNumber self, SplitComplexNumber other) {
    return split_complex_number_split_complex_number_geometric_product(self, split_complex_number_inverse(other));
}

SplitComplexNumber split_complex_number_split_complex_number_transformation(SplitComplexNumber self, SplitComplexNumber other) {
    return split_complex_number_split_complex_number_geometric_product(split_complex_number_split_complex_number_geometric_product(self, other), split_complex_number_reversal(self));
}

