/*
dynamics_solver_forcespro : A fast customized optimization solver.

Copyright (C) 2013-2025 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

#include "mex.h"
#include "math.h"
#include "../include/dynamics_solver_forcespro.h"
#include "../include/dynamics_solver_forcespro_memory.h"
#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif



/* copy functions */

void copyCArrayToM_double(double *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_double(double *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double) (*src++) ;
    }
}

void copyMValueToC_double(double * src, double * dest)
{
	*dest = (double) *src;
}/* copy functions */

void copyCArrayToM_solver_int32_default(solver_int32_default *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_solver_int32_default(double *src, solver_int32_default *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (solver_int32_default) (*src++) ;
    }
}

void copyMValueToC_solver_int32_default(double * src, solver_int32_default * dest)
{
	*dest = (solver_int32_default) *src;
}/* copy functions */

void copyCArrayToM_dynamics_solver_forcespro_float(dynamics_solver_forcespro_float *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyMArrayToC_dynamics_solver_forcespro_float(double *src, dynamics_solver_forcespro_float *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (dynamics_solver_forcespro_float) (*src++) ;
    }
}

void copyMValueToC_dynamics_solver_forcespro_float(double * src, dynamics_solver_forcespro_float * dest)
{
	*dest = (dynamics_solver_forcespro_float) *src;
}

extern solver_int32_default (dynamics_solver_forcespro_float *x, dynamics_solver_forcespro_float *y, dynamics_solver_forcespro_float *l, dynamics_solver_forcespro_float *p, dynamics_solver_forcespro_float *f, dynamics_solver_forcespro_float *nabla_f, dynamics_solver_forcespro_float *c, dynamics_solver_forcespro_float *nabla_c, dynamics_solver_forcespro_float *h, dynamics_solver_forcespro_float *nabla_h, dynamics_solver_forcespro_float *hess, solver_int32_default stage, solver_int32_default iteration, solver_int32_default threadID);
dynamics_solver_forcespro_extfunc pt2function_dynamics_solver_forcespro = &;


/* Some memory for MEX function */
static dynamics_solver_forcespro_params params;
static dynamics_solver_forcespro_output output;
static dynamics_solver_forcespro_info info;
static dynamics_solver_forcespro_mem * mem;

/* Main MEX function */
void mexFunction( solver_int32_default nlhs, mxArray *plhs[], solver_int32_default nrhs, const mxArray *prhs[] )  
{
	/* file pointer for printing */
	FILE *fp = NULL;

	/* define variables */	
	mxArray *par;
	mxArray *outvar;
	const mxArray *PARAMS = prhs[0]; 
	double *pvalue;
	solver_int32_default i;
	solver_int32_default exitflag;
	const solver_int8_default *fname;
	const solver_int8_default *outputnames[35] = {"x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34", "x35"};
	const solver_int8_default *infofields[20] = { "it", "it2opt", "res_eq", "res_ineq", "rsnorm", "rcompnorm", "pobj", "dobj", "dgap", "rdgap", "mu", "mu_aff", "sigma", "lsit_aff", "lsit_cc", "step_aff", "step_cc", "solvetime", "fevalstime", "solver_id"};
	
	/* Check for proper number of arguments */
    if (nrhs != 1)
	{
		mexErrMsgTxt("This function requires exactly 1 input: PARAMS struct.\nType 'help dynamics_solver_forcespro_mex' for details.");
	}    
	if (nlhs > 3) 
	{
        mexErrMsgTxt("This function returns at most 3 outputs.\nType 'help dynamics_solver_forcespro_mex' for details.");
    }

	/* Check whether params is actually a structure */
	if( !mxIsStruct(PARAMS) ) 
	{
		mexErrMsgTxt("PARAMS must be a structure.");
	}
	 

    /* initialize memory */
    if (mem == NULL)
    {
        mem = dynamics_solver_forcespro_internal_mem(0);
    }

	/* copy parameters into the right location */
	par = mxGetField(PARAMS, 0, "x0");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.x0 not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.x0 must be a double.");
    }
    if( mxGetM(par) != 420 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.x0 must be of size [420 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.x0,420);

	}
	par = mxGetField(PARAMS, 0, "xinit");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.xinit not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.xinit must be a double.");
    }
    if( mxGetM(par) != 9 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.xinit must be of size [9 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.xinit,9);

	}
	par = mxGetField(PARAMS, 0, "all_parameters");
#ifdef MEXARGMUENTCHECKS
    if( par == NULL )	
	{
        mexErrMsgTxt("PARAMS.all_parameters not found");
    }
    if( !mxIsDouble(par) )
    {
    mexErrMsgTxt("PARAMS.all_parameters must be a double.");
    }
    if( mxGetM(par) != 595 || mxGetN(par) != 1 ) 
	{
    mexErrMsgTxt("PARAMS.all_parameters must be of size [595 x 1]");
    }
#endif	 
	if ( (mxGetN(par) != 0) && (mxGetM(par) != 0) )
	{
		copyMArrayToC_double(mxGetPr(par), params.all_parameters,595);

	}


	#if SET_PRINTLEVEL_dynamics_solver_forcespro > 0
		/* Prepare file for printfs */
        fp = fopen("stdout_temp","w+");
		if( fp == NULL ) 
		{
			mexErrMsgTxt("freopen of stdout did not work.");
		}
		rewind(fp);
	#endif

	/* call solver */

	exitflag = dynamics_solver_forcespro_solve(&params, &output, &info, mem, fp, pt2function_dynamics_solver_forcespro);
	
	#if SET_PRINTLEVEL_dynamics_solver_forcespro > 0
		/* Read contents of printfs printed to file */
		rewind(fp);
		while( (i = fgetc(fp)) != EOF ) 
		{
			mexPrintf("%c",i);
		}
		fclose(fp);
	#endif

	/* copy output to matlab arrays */
	plhs[0] = mxCreateStructMatrix(1, 1, 35, outputnames);
		/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x01[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x01", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x02[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x02", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x03[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x03", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x04[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x04", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x05[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x05", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x06[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x06", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x07[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x07", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x08[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x08", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x09[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x09", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x10[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x10", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x11[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x11", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x12[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x12", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x13[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x13", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x14[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x14", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x15[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x15", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x16[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x16", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x17[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x17", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x18[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x18", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x19[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x19", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x20[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x20", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x21[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x21", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x22[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x22", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x23[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x23", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x24[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x24", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x25[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x25", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x26[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x26", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x27[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x27", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x28[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x28", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x29[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x29", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x30[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x30", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x31[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x31", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x32[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x32", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x33[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x33", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x34[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x34", outvar);


	/* column vector of length 12 */
	outvar = mxCreateDoubleMatrix(12, 1, mxREAL);
	copyCArrayToM_double((&(output.x35[0])), mxGetPr(outvar), 12);
	mxSetField(plhs[0], 0, "x35", outvar);


	/* copy exitflag */
	if( nlhs > 1 )
	{
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	*mxGetPr(plhs[1]) = (double)exitflag;
	}

	/* copy info struct */
	if( nlhs > 2 )
	{
	plhs[2] = mxCreateStructMatrix(1, 1, 20, infofields);
				/* scalar: iteration number */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.it)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "it", outvar);


		/* scalar: number of iterations needed to optimality (branch-and-bound) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.it2opt)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "it2opt", outvar);


		/* scalar: inf-norm of equality constraint residuals */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.res_eq)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "res_eq", outvar);


		/* scalar: inf-norm of inequality constraint residuals */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.res_ineq)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "res_ineq", outvar);


		/* scalar: norm of stationarity condition */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.rsnorm)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "rsnorm", outvar);


		/* scalar: max of all complementarity violations */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.rcompnorm)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "rcompnorm", outvar);


		/* scalar: primal objective */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.pobj)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "pobj", outvar);


		/* scalar: dual objective */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.dobj)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "dobj", outvar);


		/* scalar: duality gap := pobj - dobj */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.dgap)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "dgap", outvar);


		/* scalar: relative duality gap := |dgap / pobj | */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.rdgap)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "rdgap", outvar);


		/* scalar: duality measure */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.mu)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "mu", outvar);


		/* scalar: duality measure (after affine step) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.mu_aff)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "mu_aff", outvar);


		/* scalar: centering parameter */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.sigma)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "sigma", outvar);


		/* scalar: number of backtracking line search steps (affine direction) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.lsit_aff)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "lsit_aff", outvar);


		/* scalar: number of backtracking line search steps (combined direction) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.lsit_cc)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "lsit_cc", outvar);


		/* scalar: step size (affine direction) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.step_aff)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "step_aff", outvar);


		/* scalar: step size (combined direction) */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.step_cc)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "step_cc", outvar);


		/* scalar: total solve time */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.solvetime)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "solvetime", outvar);


		/* scalar: time spent in function evaluations */
		outvar = mxCreateDoubleMatrix(1, 1, mxREAL);
		copyCArrayToM_dynamics_solver_forcespro_float((&(info.fevalstime)), mxGetPr(outvar), 1);
		mxSetField(plhs[2], 0, "fevalstime", outvar);


		/* column vector of length 8: solver ID of FORCESPRO solver */
		outvar = mxCreateDoubleMatrix(8, 1, mxREAL);
		copyCArrayToM_solver_int32_default((&(info.solver_id[0])), mxGetPr(outvar), 8);
		mxSetField(plhs[2], 0, "solver_id", outvar);

	}
}
