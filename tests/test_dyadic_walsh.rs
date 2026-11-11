use versorc_math::prelude::*;

#[test]
fn test_fast_walsh_hadamard_unitarity_and_parseval() {
    let dim = 64;
    let seed = 42;
    let versor = DyadicWalshVersor::new(dim, seed);

    let mut vec = (0..dim).map(|i| (i as f32) * 0.1 - 3.2).collect::<Vec<_>>();
    let original = vec.clone();

    let energy_before: f32 = vec.iter().map(|&x| x * x).sum();

    // Forward dyadic versor rotation
    versor.apply_in_place(&mut vec);

    let energy_after: f32 = vec.iter().map(|&x| x * x).sum();

    // Parseval Energy Conservation
    assert!(
        (energy_after - energy_before).abs() < 1e-4,
        "Dyadic versor must preserve total L2 energy (Parseval identity), before: {}, after: {}",
        energy_before,
        energy_after
    );

    // Invert rotation: R^-1(R(x)) == x
    versor.invert_in_place(&mut vec);

    for (i, (&a, &b)) in original.iter().zip(&vec).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Dyadic inversion must recover original vector at index {}, orig: {}, recovered: {}",
            i,
            a,
            b
        );
    }
}
