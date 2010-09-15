// Functions for complex numbers in CUDA

#include <cuComplex.h>

/* versions for hosts without native support for 'complex' */
#if !defined(CU_COMPLEX_FUNCS_H_)
#define CU_COMPLEX_FUNCS_H_

#if (!defined(__CUDACC__) && defined(CU_USE_NATIVE_COMPLEX))
#include <complex.h>

/* -- Single Precision -- */

__host__ __device__ static __inline__ cuFloatComplex cuCsinf (cuFloatComplex z)
{
    return csinf(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuCcosf (cuFloatComplex z)
{
    return ccosf(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuCtanf (cuFloatComplex z)
{
    return ctanf(z);
}

__host__ __device__ static __inline__ float cuCargf (cuFloatComplex z)
{
    return cargf(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuCexpf (cuFloatComplex z)
{
    return cexpf(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuClogf (cuFloatComplex z)
{
    return clogf(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuCsqrtf (cuFloatComplex z)
{
    return csqrtf(z);
}

/* -- Double Precision -- */

__host__ __device__ static __inline__ cuDoubleComplex cuCsin (cuDoubleComplex z)
{
    return csin(z);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCcos (cuDoubleComplex z)
{
    return ccos(z);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCtan (cuDoubleComplex z)
{
    return ctan(z);
}

__host__ __device__ static __inline__ double cuCarg (cuDoubleComplex z)
{
    return carg(z);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCexp (cuDoubleComplex z)
{
    return cexp(z);
}

__host__ __device__ static __inline__ cuDoubleComplex cuClog (cuDoubleComplex z)
{
    return clog(z);
}

__host__ __device__ static __inline__ cuFloatComplex cuCsqrt (cuFloatComplex z)
{
    return csqrt(z);
}

/* versions for target or hosts without native support for 'complex' */

#else /* (!defined(__CUDACC__) && defined(CU_USE_NATIVE_COMPLEX)) */

/* -- Single precision -- */

__host__ __device__ static __inline__ cuFloatComplex cuCsinf (cuFloatComplex z)
{
    float a = expf(z.y);
    float b = expf(-z.y);
    float sin_x, cos_x;
    sincosf(z.x, &sin_x, &cos_x);
    return make_cuFloatComplex((sin_x/2)*(a+b), (cos_x/2)*(a-b));
}

__host__ __device__ static __inline__ cuFloatComplex cuCcosf (cuFloatComplex z)
{
    float a = expf(z.y);
    float b = expf(-z.y);
    float sin_x, cos_x;
    sincosf(z.x, &sin_x, &cos_x);
    return make_cuFloatComplex((cos_x/2)*(b+a), (sin_x/2)*(b-a));
}

__host__ __device__ static __inline__ cuFloatComplex cuCtanf (cuFloatComplex z)
{
    return cuCdivf(cuCsinf(z), cuCcosf(z));
}

__host__ __device__ static __inline__ float cuCargf (cuFloatComplex z)
{
    return atan2f(z.y, z.x);
}

__host__ __device__ static __inline__ cuFloatComplex cuCexpf (cuFloatComplex z)
{
    float a = expf(z.x);
    float sin_y, cos_y;
    sincosf(z.y, &sin_y, &cos_y);
    return make_cuFloatComplex(a*cos_y, a*sin_y);
}

__host__ __device__ static __inline__ cuFloatComplex cuClogf (cuFloatComplex z)
{
    return make_cuFloatComplex(logf(cuCabsf(z)), cuCargf(z));
}

__host__ __device__ static __inline__ cuFloatComplex cuCsqrtf (cuFloatComplex z)
{
    float a = powf(z.x*z.x+z.y*z.y, 1.0/4.0);
    float sin_atan2_yx, cos_atan2_yx;
    sincosf(atan2f(z.y, z.x)/2.0, &sin_atan2_yx, &cos_atan2_yx);
    return make_cuFloatComplex(a*cos_atan2_yx, a*sin_atan2_yx);
}

/* -- Double precision -- */

__host__ __device__ static __inline__ cuDoubleComplex cuCsin (cuDoubleComplex z)
{
    float a = exp(z.y);
    float b = exp(-z.y);
    float sin_x, cos_x;
    sincos(z.x, &sin_x, &cos_x);
    return make_cuDoubleComplex((sin_x/2)*(a+b), (cos_x/2)*(a-b));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCcos (cuDoubleComplex z)
{
    float a = exp(z.y);
    float b = exp(-z.y);
    float sin_x, cos_x;
    sincos(z.x, &sin_x, &cos_x);
    return make_cuDoubleComplex((cos_x/2)*(b+a), (sin_x/2)*(b-a));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCtan (cuDoubleComplex z)
{
    return cuCdiv(cuCsin(z), cuCcos(z));
}

__host__ __device__ static __inline__ double cuCarg (cuDoubleComplex z)
{
    return atan2(z.y, z.x);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCexp (cuDoubleComplex z)
{
    float a = exp(z.x);
    float sin_y, cos_y;
    sincos(z.y, &sin_y, &cos_y);
    return make_cuDoubleComplex(a*cos_y, a*sin_y);
}

__host__ __device__ static __inline__ cuDoubleComplex cuClog (cuDoubleComplex z)
{
    return make_cuDoubleComplex(log(cuCabs(z)), cuCarg(z));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCsqrt (cuDoubleComplex z)
{
    float a = pow(z.x*z.x+z.y*z.y, 1.0/4.0);
    float sin_atan2_yx, cos_atan2_yx;
    sincos(atan2(z.y, z.x)/2.0, &sin_atan2_yx, &cos_atan2_yx);
    return make_cuDoubleComplex(a*cos_atan2_yx, a*sin_atan2_yx);
}

#endif /* (!defined(__CUDACC__) && defined(CU_USE_NATIVE_COMPLEX)) */

#endif /* !defined(CU_COMPLEX_FUNCS) */
