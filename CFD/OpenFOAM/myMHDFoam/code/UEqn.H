// Momentum equation setup with Lorentz force
fvVectorMatrix UEqn
(
    fvm::ddt(U) 
  + fvm::div(phi, U) 
  - fvm::laplacian(nu, U)
);

UEqn() == -fvc::grad(p) + rhoE * E + fvc::curl(B) ^ J; // Including Lorentz force

// Solve the momentum equation
solve(UEqn == -fvc::grad(p));
