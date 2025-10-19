/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2025. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "include/dynamics_solver_forcespro_drl.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "dynamics_solver_forcespro_drl_model.h"



/* copies data from sparse matrix into a dense one */
static void dynamics_solver_forcespro_drl_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, dynamics_solver_forcespro_drl_callback_float *data, dynamics_solver_forcespro_drl_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((dynamics_solver_forcespro_drl_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default dynamics_solver_forcespro_drl_adtool2forces(dynamics_solver_forcespro_drl_float *x,        /* primal vars                                         */
                                 dynamics_solver_forcespro_drl_float *y,        /* eq. constraint multiplers                           */
                                 dynamics_solver_forcespro_drl_float *l,        /* ineq. constraint multipliers                        */
                                 dynamics_solver_forcespro_drl_float *p,        /* parameters                                          */
                                 dynamics_solver_forcespro_drl_float *f,        /* objective function (scalar)                         */
                                 dynamics_solver_forcespro_drl_float *nabla_f,  /* gradient of objective function                      */
                                 dynamics_solver_forcespro_drl_float *c,        /* dynamics                                            */
                                 dynamics_solver_forcespro_drl_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 dynamics_solver_forcespro_drl_float *h,        /* inequality constraints                              */
                                 dynamics_solver_forcespro_drl_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 dynamics_solver_forcespro_drl_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
                                 solver_int32_default iteration, /* iteration number of solver                         */
                                 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const dynamics_solver_forcespro_drl_callback_float *in[4];
    dynamics_solver_forcespro_drl_callback_float *out[7];
    

    /* Allocate working arrays for AD tool */
    
    dynamics_solver_forcespro_drl_callback_float w[37];
	
    /* temporary storage for AD tool sparse output */
    dynamics_solver_forcespro_drl_callback_float this_f = (dynamics_solver_forcespro_drl_callback_float) 0.0;
    dynamics_solver_forcespro_drl_float nabla_f_sparse[8];
    dynamics_solver_forcespro_drl_float h_sparse[2];
    dynamics_solver_forcespro_drl_float nabla_h_sparse[5];
    dynamics_solver_forcespro_drl_float c_sparse[1];
    dynamics_solver_forcespro_drl_float nabla_c_sparse[1];
    
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 33))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		dynamics_solver_forcespro_drl_objective_0(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_objective_0_sparsity_out(1)[0];
			ncol = dynamics_solver_forcespro_drl_objective_0_sparsity_out(1)[1];
			colind = dynamics_solver_forcespro_drl_objective_0_sparsity_out(1) + 2;
			row = dynamics_solver_forcespro_drl_objective_0_sparsity_out(1) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		dynamics_solver_forcespro_drl_rkfour_0(x, p, c, nabla_c, dynamics_solver_forcespro_drl_cdyn_0rd_0, dynamics_solver_forcespro_drl_cdyn_0, threadID);
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		dynamics_solver_forcespro_drl_inequalities_0(in, out, NULL, w, 0);
		if( h != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(0)[0];
			ncol = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(0)[1];
			colind = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(0) + 2;
			row = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(1)[0];
			ncol = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(1)[1];
			colind = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(1) + 2;
			row = dynamics_solver_forcespro_drl_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((34 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		dynamics_solver_forcespro_drl_objective_1(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_objective_1_sparsity_out(1)[0];
			ncol = dynamics_solver_forcespro_drl_objective_1_sparsity_out(1)[1];
			colind = dynamics_solver_forcespro_drl_objective_1_sparsity_out(1) + 2;
			row = dynamics_solver_forcespro_drl_objective_1_sparsity_out(1) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		dynamics_solver_forcespro_drl_inequalities_1(in, out, NULL, w, 0);
		if( h != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(0)[0];
			ncol = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(0)[1];
			colind = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(0) + 2;
			row = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h != NULL )
		{
			nrow = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(1)[0];
			ncol = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(1)[1];
			colind = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(1) + 2;
			row = dynamics_solver_forcespro_drl_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
				
			dynamics_solver_forcespro_drl_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((dynamics_solver_forcespro_drl_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
