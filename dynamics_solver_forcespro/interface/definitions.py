import numpy
import ctypes

name = "dynamics_solver_forcespro"
requires_callback = True
lib = "lib/libdynamics_solver_forcespro.so"
lib_static = "lib/libdynamics_solver_forcespro.a"
c_header = "include/dynamics_solver_forcespro.h"
nstages = 35

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (420,   1),  420),
 ("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  9,   1),    9),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (595,   1),  595)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x11"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x12"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x13"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x14"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x15"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x16"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x17"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x18"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x19"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x20"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x21"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x22"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x23"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x24"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x25"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x26"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x27"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x28"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x29"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x30"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x31"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x32"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x33"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x34"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12),
 ("x35"                 , ""               , ctypes.c_double, numpy.float64,     ( 12,),   12)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(12, 9, 2, 17, 3, 3, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 9, 2, 17, 12, 12, 2, 2), 
	(12, 0, 2, 17, 12, 12, 2, 2)
]