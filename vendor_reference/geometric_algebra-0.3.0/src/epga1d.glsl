struct ComplexNumber {
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

ComplexNumber scalar_complex_number_add(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * vec2(1.0, 0.0) + other.g0);
}

ComplexNumber scalar_complex_number_sub(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * vec2(1.0, 0.0) - other.g0);
}

ComplexNumber scalar_complex_number_geometric_product(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * other.g0);
}

float scalar_complex_number_regressive_product(float self, ComplexNumber other) {
    return self * other.g0.y;
}

ComplexNumber scalar_complex_number_outer_product(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * other.g0);
}

ComplexNumber scalar_complex_number_inner_product(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * other.g0);
}

ComplexNumber scalar_complex_number_left_contraction(float self, ComplexNumber other) {
    return ComplexNumber(vec2(self) * other.g0);
}

float scalar_complex_number_right_contraction(float self, ComplexNumber other) {
    return self * other.g0.x;
}

float scalar_complex_number_scalar_product(float self, ComplexNumber other) {
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

ComplexNumber complex_number_zero() {
    return ComplexNumber(vec2(0.0));
}

ComplexNumber complex_number_one() {
    return ComplexNumber(vec2(1.0, 0.0));
}

ComplexNumber complex_number_neg(ComplexNumber self) {
    return ComplexNumber(self.g0 * vec2(-1.0));
}

ComplexNumber complex_number_automorphism(ComplexNumber self) {
    return ComplexNumber(self.g0);
}

ComplexNumber complex_number_reversal(ComplexNumber self) {
    return ComplexNumber(self.g0 * vec2(1.0, -1.0));
}

ComplexNumber complex_number_conjugation(ComplexNumber self) {
    return ComplexNumber(self.g0 * vec2(1.0, -1.0));
}

ComplexNumber complex_number_dual(ComplexNumber self) {
    return ComplexNumber(self.g0.yx);
}

float complex_number_scalar_into(ComplexNumber self) {
    return self.g0.x;
}

ComplexNumber complex_number_scalar_add(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 + vec2(other) * vec2(1.0, 0.0));
}

ComplexNumber complex_number_scalar_sub(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 - vec2(other) * vec2(1.0, 0.0));
}

ComplexNumber complex_number_scalar_geometric_product(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 * vec2(other));
}

float complex_number_scalar_regressive_product(ComplexNumber self, float other) {
    return self.g0.y * other;
}

ComplexNumber complex_number_scalar_outer_product(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 * vec2(other));
}

ComplexNumber complex_number_scalar_inner_product(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 * vec2(other));
}

float complex_number_scalar_left_contraction(ComplexNumber self, float other) {
    return self.g0.x * other;
}

ComplexNumber complex_number_scalar_right_contraction(ComplexNumber self, float other) {
    return ComplexNumber(self.g0 * vec2(other));
}

float complex_number_scalar_scalar_product(ComplexNumber self, float other) {
    return self.g0.x * other;
}

ComplexNumber complex_number_complex_number_add(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(self.g0 + other.g0);
}

ComplexNumber complex_number_complex_number_sub(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(self.g0 - other.g0);
}

ComplexNumber complex_number_complex_number_mul(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(self.g0 * other.g0);
}

ComplexNumber complex_number_complex_number_div(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.x, self.g0.y) * vec2(1.0, 1.0) / vec2(other.g0.x, other.g0.y) * vec2(1.0, 1.0));
}

ComplexNumber complex_number_complex_number_geometric_product(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.x) * other.g0 + vec2(self.g0.y) * other.g0.yx * vec2(-1.0, 1.0));
}

ComplexNumber complex_number_complex_number_regressive_product(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.y) * other.g0 + vec2(self.g0.x) * other.g0.yx * vec2(1.0, 0.0));
}

ComplexNumber complex_number_complex_number_outer_product(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.x) * other.g0 + self.g0 * vec2(other.g0.x) * vec2(0.0, 1.0));
}

ComplexNumber complex_number_complex_number_inner_product(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.x) * other.g0 + vec2(self.g0.y) * other.g0.yx * vec2(-1.0, 1.0));
}

ComplexNumber complex_number_complex_number_left_contraction(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.x) * other.g0 + self.g0.yx * other.g0.yx * vec2(-1.0, 0.0));
}

ComplexNumber complex_number_complex_number_right_contraction(ComplexNumber self, ComplexNumber other) {
    return ComplexNumber(vec2(self.g0.y) * other.g0.yx * vec2(-1.0, 1.0) + vec2(self.g0.x) * vec2(other.g0.x) * vec2(1.0, 0.0));
}

float complex_number_complex_number_scalar_product(ComplexNumber self, ComplexNumber other) {
    return self.g0.x * other.g0.x - self.g0.y * other.g0.y;
}

float complex_number_squared_magnitude(ComplexNumber self) {
    return complex_number_complex_number_scalar_product(self, complex_number_reversal(self));
}

float complex_number_magnitude(ComplexNumber self) {
    return sqrt(complex_number_squared_magnitude(self));
}

ComplexNumber complex_number_scalar_mul(ComplexNumber self, float other) {
    return complex_number_scalar_geometric_product(self, other);
}

ComplexNumber complex_number_signum(ComplexNumber self) {
    return complex_number_scalar_geometric_product(self, 1.0 / complex_number_magnitude(self));
}

ComplexNumber complex_number_inverse(ComplexNumber self) {
    return complex_number_scalar_geometric_product(complex_number_reversal(self), 1.0 / complex_number_squared_magnitude(self));
}

ComplexNumber complex_number_powi(ComplexNumber self, int exponent) {
    if(exponent == 0) {
        return complex_number_one();
    }
    ComplexNumber x = (exponent < 0) ? complex_number_inverse(self) : self;
    ComplexNumber y = complex_number_one();
    int n = abs(exponent);
    while(1 < n) {
        if((n & 1) == 1) {
            y = complex_number_complex_number_geometric_product(x, y);
        }
        x = complex_number_complex_number_geometric_product(x, x);
        n = n >> 1;
    }
    return complex_number_complex_number_geometric_product(x, y);
}

ComplexNumber complex_number_complex_number_geometric_quotient(ComplexNumber self, ComplexNumber other) {
    return complex_number_complex_number_geometric_product(self, complex_number_inverse(other));
}

ComplexNumber complex_number_complex_number_transformation(ComplexNumber self, ComplexNumber other) {
    return complex_number_complex_number_geometric_product(complex_number_complex_number_geometric_product(self, other), complex_number_reversal(self));
}

ComplexNumber complex_number_scalar_geometric_quotient(ComplexNumber self, float other) {
    return complex_number_scalar_geometric_product(self, scalar_inverse(other));
}

float complex_number_scalar_transformation(ComplexNumber self, float other) {
    return complex_number_scalar_into(complex_number_complex_number_geometric_product(complex_number_scalar_geometric_product(self, other), complex_number_reversal(self)));
}

ComplexNumber scalar_complex_number_geometric_quotient(float self, ComplexNumber other) {
    return scalar_complex_number_geometric_product(self, complex_number_inverse(other));
}

ComplexNumber scalar_complex_number_transformation(float self, ComplexNumber other) {
    return complex_number_scalar_geometric_product(scalar_complex_number_geometric_product(self, other), scalar_reversal(self));
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

