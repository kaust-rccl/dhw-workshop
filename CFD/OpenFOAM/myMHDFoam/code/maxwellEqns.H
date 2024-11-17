// Ampere's Law with Maxwell correction
fvVectorMatrix MaxwellEqn
(
    fvm::curl(B) == mu0 * (J + epsilon0 * fvc::ddt(E))
);
solve(MaxwellEqn);

// Faraday’s Law of Induction
fvVectorMatrix FaradayEqn
(
    fvm::curl(E) == -fvc::ddt(B)
);
solve(FaradayEqn);
