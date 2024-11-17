// Solve the pressure equation in a PISO loop

// Compute the flux for the pressure equation
phi = fvc::interpolate(U) & mesh.Sf();

// Adjust flux to be consistent with the current velocity field
surfaceScalarField rAUf = 1.0 / UEqn.A();
phi -= fvc::div(phi) * rAUf * fvc::snGrad(p) * mesh.magSf();

// Pressure equation
fvScalarMatrix pEqn
(
    fvm::laplacian(rAUf, p) == fvc::div(phi)
);

// Apply boundary conditions to pressure field
pEqn.solve();

// Correct the flux to account for the new pressure
phi = phi - pEqn.flux();

// Correct the velocity field using the updated pressure gradient
U += rAUf * fvc::grad(p);
U.correctBoundaryConditions();
