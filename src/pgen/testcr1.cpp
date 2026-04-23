//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min

// Athena++ headers
#include "../globals.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../cr/cr.hpp"
#include "../cr/integrators/cr_integrators.hpp"


//======================================================================================
/*! \file Test case 1 in Jiang and Oh 2018
 *  \brief Simple streaming test
 *
 *====================================================================================*/


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================

static Real gg = 5./3.;
static Real gg1 = gg/(gg - 1.);
static Real gc = 4./3.;
static Real gc1 = gc/(gc - 1.);

// Sigma for diffusion, need to multiply by reduced speed of light to get correct code units
static Real sigma_diff = 1.e10; // For vmax=100

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void Mesh::UserWorkAfterLoop(ParameterInput *pin) { 
    
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED){
    pcr->EnrollOpacityFunction(Diffusion);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);

        Real dist_sq = x1*x1;
      
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS){
          phydro->u(IEN,k,j,i) = 1.0/(gg-1.0);
        }
        
        if (CR_ENABLED) {
          Real Ec = std::exp(-40.0 * dist_sq);
          Real Pc = Ec / 3.0;

          // analytic dPc/dx for Ec = exp(-40 x^2)
          Real dPcdx = -(80.0/3.0) * x1 * Ec;

          // smooth sign, matching the spirit of DefaultStreaming
          const Real f_smooth_ic = 1.0e-1;
          Real L_box = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
          Real grad_floor = f_smooth_ic * std::max(std::abs(dPcdx), Pc / std::max(L_box, TINY_NUMBER));
          Real s = dPcdx / std::sqrt(dPcdx*dPcdx + grad_floor*grad_floor + TINY_NUMBER);

          // here v_A = B/sqrt(rho) = 1
          Real vstream = -1.0 * s;

          pcr->u_cr(CRE,k,j,i)  = Ec;
          //pcr->u_cr(CRF1,k,j,i) = (Ec + Pc) * vstream / pcr->vmax;
          pcr->u_cr(CRF1,k,j,i) = 0.0;
          pcr->u_cr(CRF2,k,j,i) = 0.0;
          pcr->u_cr(CRF3,k,j,i) = 0.0;
        }
      }// end i
    }
  }
  //Need to set opactiy sigma in the ghost zones
  if(CR_ENABLED){

  // Default values are 1/3
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if(nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if(nz3 > 1) nz3 += 2*(NGHOST);
    for(int k=0; k<nz3; ++k){
      for(int j=0; j<nz2; ++j){
        for(int i=0; i<nz1; ++i){
          pcr->sigma_diff(0,k,j,i) = sigma_diff;
          pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
        }
      }
    }// end k,j,i

  }// End CR

    // Add horizontal magnetic field lines, to show streaming and diffusion 
  // along magnetic field ines
  if(MAGNETIC_FIELDS_ENABLED){

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 1.0;
        }
      }
    }

    if(block_size.nx2 > 1){

      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }

    }

    if(block_size.nx3 > 1){

      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
    }// end nx3

    // set cell centerd magnetic field
    // Add magnetic energy density to the total energy
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        for(int i=is; i<=ie; ++i){
          phydro->u(IEN,k,j,i) +=
            0.5*(SQR((pfield->bcc(IB1,k,j,i)))
               + SQR((pfield->bcc(IB2,k,j,i)))
               + SQR((pfield->bcc(IB3,k,j,i))));
      
        }
      }
    }

  }// end MHD
  
  
  return;
}

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
               AthenaArray<Real> &prim, AthenaArray<Real> &bcc) {
  CosmicRay *pcr = pmb->pcr;
  int kl = pmb->ks, ku = pmb->ke;
  int jl = pmb->js, ju = pmb->je;
  int il = pmb->is - 1, iu = pmb->ie + 1;
  if (pmb->block_size.nx2 > 1) { jl -= 1; ju += 1; }
  if (pmb->block_size.nx3 > 1) { kl -= 1; ku += 1; }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        pcr->sigma_diff(0,k,j,i) = sigma_diff;
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          Real bx = bcc(IB1,k,j,i);
          Real by = bcc(IB2,k,j,i);
          Real bz = bcc(IB3,k,j,i);
          Real bxby = std::sqrt(bx*bx + by*by);
          Real btot = std::sqrt(bx*bx + by*by + bz*bz);

          if (btot > TINY_NUMBER) {
            pcr->b_angle(0,k,j,i) = bxby / btot;
            pcr->b_angle(1,k,j,i) = bz / btot;
          } else {
            pcr->b_angle(0,k,j,i) = 1.0;
            pcr->b_angle(1,k,j,i) = 0.0;
          }

          if (bxby > TINY_NUMBER) {
            pcr->b_angle(2,k,j,i) = by / bxby;
            pcr->b_angle(3,k,j,i) = bx / bxby;
          } else {
            pcr->b_angle(2,k,j,i) = 0.0;
            pcr->b_angle(3,k,j,i) = 1.0;
          }

          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
        }
      }
    }
  } else {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
        }
      }
    }
  }
}