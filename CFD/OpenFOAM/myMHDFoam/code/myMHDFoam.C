/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    Navier Stroke Equation Coupled with Maxwell eq

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, laminar flow of Newtonian fluids.

    \heading Solver details
    The solver uses the PISO algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \ddt{\vec{U}}
          + \div \left( \vec{U} \vec{U} \right)
          - \div \left(\nu \grad \vec{U} \right)
          = - \grad p
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
    \endvartable

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H" // Include the core OpenFOAM header
#include "createFields.H"
#include "UEqn.H"
#include "pEqn.H"
#include "maxwellEqns.H" // Include Maxwell equations

int main(int argc, char *argv[])
{
    // Initialize the OpenFOAM framework
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    // Create fields
    #include "createFields.H"
    
    while (runTime.run())
    {
        // Time step management
        #include "readTimeControls.H"
        runTime++;
        
        // Momentum equation
        #include "UEqn.H"
        
        // Pressure equation
        #include "pEqn.H"
        
        // Electromagnetic equations
        #include "maxwellEqns.H"
        
        // Write results
        runTime.write();
    }
    
    return 0;
}


// ************************************************************************* //
