//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file cr.cpp
//  \brief implementation of functions in cosmic ray class
//======================================================================================

// C headers

// C++ headers
#include <cstdio>  // fopen and fwrite
#include <iostream>  // cout
#include <sstream>  // msg
#include <stdexcept> // runtime erro

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "cr.hpp"
#include "integrators/cr_integrators.hpp"

#include <fstream>
#include <iomanip>

// constructor, initializes data structures and parameters

// The default opacity function.

// This function also needs to set the streaming velocity
// This is needed to calculate the work term
inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_cr,
              AthenaArray<Real> &prim, AthenaArray<Real> &bcc) {
  // set the default opacity to be a large value in the default hydro case
  CosmicRay *pcr=pmb->pcr;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if (pmb->block_size.nx2 > 1) {
    jl -= 1;
    ju += 1;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= 1;
    ku += 1;
  }
  Real invlim = 1.0/pcr->vmax;
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        pcr->sigma_diff(0,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
      }
    }
  }

  // Need to calculate the rotation matrix
  // We need this to determine the direction of rotation velocity
  if (MAGNETIC_FIELDS_ENABLED && (pcr->stream_flag > 0)) {
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

          // In the new closure, sigma_adv is disabled everywhere.
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

          pcr->v_adv(0,k,j,i) = 0.0;
          pcr->v_adv(1,k,j,i) = 0.0;
          pcr->v_adv(2,k,j,i) = 0.0;
        }
      }
    }
  }
}

// inline void DefaultStreaming(MeshBlock *pmb, AthenaArray<Real> &u_cr,
//              AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
//              AthenaArray<Real> &grad_pc, int k, int j, int is, int ie) {
//   CosmicRay *pcr = pmb->pcr;

//   const Real f_smooth = 3.0e-1;

//   Real L_box = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
//   if (pmb->block_size.nx2 > 1) {
//     L_box = std::min(L_box,
//       pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min);
//   }
//   if (pmb->block_size.nx3 > 1) {
//     L_box = std::min(L_box,
//       pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min);
//   }
//   L_box = std::max(L_box, TINY_NUMBER);

//   for (int i = is; i <= ie; ++i) {
//     Real rho = std::max(prim(IDN,k,j,i), TINY_NUMBER);
//     Real inv_sqrt_rho = 1.0 / std::sqrt(rho);

//     Real b1 = bcc(IB1,k,j,i);
//     Real b2 = bcc(IB2,k,j,i);
//     Real b3 = bcc(IB3,k,j,i);

//     Real b_grad_pc = b1 * grad_pc(0,k,j,i)
//                    + b2 * grad_pc(1,k,j,i)
//                    + b3 * grad_pc(2,k,j,i);

//     Real va1 = b1 * inv_sqrt_rho;
//     Real va2 = b2 * inv_sqrt_rho;
//     Real va3 = b3 * inv_sqrt_rho;

//     Real pc_local = std::max(u_cr(CRE,k,j,i) / 3.0, TINY_NUMBER);
//     Real grad_abs = std::abs(b_grad_pc);

//     Real grad_phys_floor = pc_local / L_box;
//     Real eps_grad = f_smooth * std::max(grad_abs, grad_phys_floor);

//     Real s = b_grad_pc /
//              std::sqrt(b_grad_pc * b_grad_pc + eps_grad * eps_grad + TINY_NUMBER);

//     if (pcr->stream_flag > 0) {
//       pcr->v_adv(0,k,j,i) = -va1 * s;
//       pcr->v_adv(1,k,j,i) = -va2 * s;
//       pcr->v_adv(2,k,j,i) = -va3 * s;
//     } else {
//       pcr->v_adv(0,k,j,i) = 0.0;
//       pcr->v_adv(1,k,j,i) = 0.0;
//       pcr->v_adv(2,k,j,i) = 0.0;
//     }

//     pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
//     pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
//     pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
//   }
// }

inline void DefaultStreaming(MeshBlock *pmb, AthenaArray<Real> &u_cr,
             AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
             AthenaArray<Real> &grad_pc, int k, int j, int is, int ie) {
  CosmicRay *pcr = pmb->pcr;

  // This should be small enough to recover nearly hard-sign behavior
  // away from the extremum, but nonzero to regularize b·grad Pc ~ 0.
  const Real f_smooth = 1.0e-1;

  Real L_box = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  if (pmb->block_size.nx2 > 1) {
    L_box = std::min(L_box,
      pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min);
  }
  if (pmb->block_size.nx3 > 1) {
    L_box = std::min(L_box,
      pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min);
  }
  L_box = std::max(L_box, TINY_NUMBER);

  for (int i = is; i <= ie; ++i) {
    Real rho = std::max(prim(IDN,k,j,i), TINY_NUMBER);
    Real inv_sqrt_rho = 1.0 / std::sqrt(rho);

    Real b1 = bcc(IB1,k,j,i);
    Real b2 = bcc(IB2,k,j,i);
    Real b3 = bcc(IB3,k,j,i);

    Real b_grad_pc = b1 * grad_pc(0,k,j,i)
                   + b2 * grad_pc(1,k,j,i)
                   + b3 * grad_pc(2,k,j,i);

    Real va1 = b1 * inv_sqrt_rho;
    Real va2 = b2 * inv_sqrt_rho;
    Real va3 = b3 * inv_sqrt_rho;

    Real pc_local = std::max(u_cr(CRE,k,j,i) / 3.0, TINY_NUMBER);

    // IMPORTANT:
    // Use a floor independent of |b·grad Pc| itself.
    // This means the regularization only matters near the sign flip.
    Real grad_floor = f_smooth * pc_local / L_box;

    Real s = b_grad_pc /
             std::sqrt(b_grad_pc * b_grad_pc +
                       grad_floor * grad_floor +
                       TINY_NUMBER);

      if (k == pmb->ks && j == pmb->js) {
        int ic = (is + ie) / 2;
        if (std::abs(i - ic) <= 8) {
          std::ofstream dbg("streaming_debug.txt", std::ios::app);
          dbg << std::setprecision(16);
          dbg << "i=" << i
              << " x=" << pmb->pcoord->x1v(i)
              << " Ec=" << u_cr(CRE,k,j,i)
              << " b_grad_pc=" << b_grad_pc
              << " grad_floor=" << grad_floor
              << " s=" << s
              << " va1=" << va1
              << " vadv1=" << (-va1 * s)
              << "\n";
      }
    }

    if (pcr->stream_flag > 0) {
      pcr->v_adv(0,k,j,i) = -va1 * s;
      pcr->v_adv(1,k,j,i) = -va2 * s;
      pcr->v_adv(2,k,j,i) = -va3 * s;
    } else {
      pcr->v_adv(0,k,j,i) = 0.0;
      pcr->v_adv(1,k,j,i) = 0.0;
      pcr->v_adv(2,k,j,i) = 0.0;
    }

    // Streaming no longer enters through sigma_adv in your modified closure.
    pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
    pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
    pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
  }
}

CosmicRay::CosmicRay(MeshBlock *pmb, ParameterInput *pin):
    pmy_block(pmb), u_cr(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    u_cr1(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    coarse_cr_(NCR,pmb->ncc3, pmb->ncc2, pmb->ncc1,
               (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                AthenaArray<Real>::DataStatus::empty)),
    sigma_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    // constructor overload resolution of non-aggregate class type AthenaArray<Real>
    flux{ {NCR, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
      {NCR,pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
       (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)},
      {NCR,pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
       (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)}
    },
    cr_bvar(pmb, &u_cr, &coarse_cr_, flux, true),
    UserSourceTerm_{} {
  Mesh *pm = pmy_block->pmy_mesh;
  pmb->RegisterMeshBlockData(u_cr);
  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u_cr, &coarse_cr_);
  }
  cr_source_defined = false;
  cr_bvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&cr_bvar);
  pmb->pbval->bvars_main_int.push_back(&cr_bvar);

  vmax = pin->GetOrAddReal("cr", "vmax", 1.0);
  vlim = pin->GetOrAddReal("cr", "vlim", 0.9);
  max_opacity = pin->GetOrAddReal("cr", "max_opacity", BIG_NUMBER);
  stream_flag = pin->GetOrAddInteger("cr", "vs_flag", 1);
  src_flag = pin->GetOrAddInteger("cr", "src_flag", 1);

  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  b_grad_pc.NewAthenaArray(nc3, nc2, nc1);
  b_angle.NewAthenaArray(4, nc3, nc2, nc1);

  cwidth.NewAthenaArray(nc1);
  cwidth1.NewAthenaArray(nc1);
  cwidth2.NewAthenaArray(nc1);

  // set a default opacity function
  UpdateOpacity = DefaultOpacity;
  UpdateStreaming = DefaultStreaming;
  pcrintegrator = new CRIntegrator(this, pin);
}

void CosmicRay::EnrollOpacityFunction(CROpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction;
}

void CosmicRay::EnrollStreamingFunction(CRStreamingFunc MyStreamingFunction) {
  UpdateStreaming = MyStreamingFunction;
}

void CosmicRay::EnrollUserCRSource(CRSrcTermFunc my_func) {
  UserSourceTerm_ = my_func;
  cr_source_defined = true;
}