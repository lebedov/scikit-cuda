// Special functions for CUDA

// Some of these functions are adapted from the Cephes and specfun
// libraries included in scipy:
// http://www.netlib.org/cephes/
// http://www.netlib.org/specfun/

#include <pycuda-complex.hpp>
#include "cuConstants.h"

#if !defined(CU_SPECIAL_FUNCS_H_)
#define CU_SPECIAL_FUNCS_H_

#define CFLOAT pycuda::complex<float>
#define CDOUBLE pycuda::complex<double>

/* Sinc function. */
__device__ float sincf(float x) {
    if (x == 0.0) 
	return 1.0;
    else
	return sinpif(x)/(PI*x);
}

__device__ double sinc(double x) {
    if (x == 0.0) 
	return 1.0;
    else
	return sinpi(x)/(PI*x);
}

/* Polynomial evaluation. */
__device__ float polevlf(float x, float *coef, int N) {
    float ans;
    float *p;
    int i;

    p = coef;
    ans = *p++;
    i = N;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

__device__ float p1evlf(float x, float *coef, int N) {
    float ans;
    float *p;
    int i;

    p = coef;
    ans = x + *p++;
    i = N-1;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

__device__ double polevl(double x, double *coef, int N) {
    double ans;
    double *p;
    int i;

    p = coef;
    ans = *p++;
    i = N;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

__device__ double p1evl(double x, double *coef, int N) {
    double ans;
    double *p;
    int i;

    p = coef;
    ans = x + *p++;
    i = N-1;

    do
        ans = ans*x + *p++;
    while (--i);

    return (ans);
}

/* Constants used to compute the sine/cosine integrals. */
__constant__ float SNf[] = {
    -8.39167827910303881427E-11,
    4.62591714427012837309E-8,
    -9.75759303843632795789E-6,
    9.76945438170435310816E-4,
    -4.13470316229406538752E-2,
    1.00000000000000000302E0,
};
__constant__ float SDf[] = {
    2.03269266195951942049E-12,
    1.27997891179943299903E-9,
    4.41827842801218905784E-7,
    9.96412122043875552487E-5,
    1.42085239326149893930E-2,
    9.99999999999999996984E-1,
};
__constant__ float CNf[] = {
    2.02524002389102268789E-11,
    -1.35249504915790756375E-8,
    3.59325051419993077021E-6,
    -4.74007206873407909465E-4,
    2.89159652607555242092E-2,
    -1.00000000000000000080E0,
};
__constant__ float CDf[] = {
    4.07746040061880559506E-12,
    3.06780997581887812692E-9,
    1.23210355685883423679E-6,
    3.17442024775032769882E-4,
    5.10028056236446052392E-2,
    4.00000000000000000080E0,
};
__constant__ float FN4f[] = {
    4.23612862892216586994E0,
    5.45937717161812843388E0,
    1.62083287701538329132E0,
    1.67006611831323023771E-1,
    6.81020132472518137426E-3,
    1.08936580650328664411E-4,
    5.48900223421373614008E-7,
};
__constant__ float FD4f[] = {
    /*  1.00000000000000000000E0,*/
    8.16496634205391016773E0,
    7.30828822505564552187E0,
    1.86792257950184183883E0,
    1.78792052963149907262E-1,
    7.01710668322789753610E-3,
    1.10034357153915731354E-4,
    5.48900252756255700982E-7,
};
__constant__ float FN8f[] = {
    4.55880873470465315206E-1,
    7.13715274100146711374E-1,
    1.60300158222319456320E-1,
    1.16064229408124407915E-2,
    3.49556442447859055605E-4,
    4.86215430826454749482E-6,
    3.20092790091004902806E-8,
    9.41779576128512936592E-11,
    9.70507110881952024631E-14,
};
__constant__ float FD8f[] = {
    /*  1.00000000000000000000E0,*/
    9.17463611873684053703E-1,
    1.78685545332074536321E-1,
    1.22253594771971293032E-2,
    3.58696481881851580297E-4,
    4.92435064317881464393E-6,
    3.21956939101046018377E-8,
    9.43720590350276732376E-11,
    9.70507110881952025725E-14,
};
                                  
__constant__ float GN4f[] = {
    8.71001698973114191777E-2,
    6.11379109952219284151E-1,
    3.97180296392337498885E-1,
    7.48527737628469092119E-2,
    5.38868681462177273157E-3,
    1.61999794598934024525E-4,
    1.97963874140963632189E-6,
    7.82579040744090311069E-9,
};
__constant__ float GD4f[] = {
    /*  1.00000000000000000000E0,*/
    1.64402202413355338886E0,
    6.66296701268987968381E-1,
    9.88771761277688796203E-2,
    6.22396345441768420760E-3,
    1.73221081474177119497E-4,
    2.02659182086343991969E-6,
    7.82579218933534490868E-9,
};
__constant__ float GN8f[] = {
    6.97359953443276214934E-1,
    3.30410979305632063225E-1,
    3.84878767649974295920E-2,
    1.71718239052347903558E-3,
    3.48941165502279436777E-5,
    3.47131167084116673800E-7,
    1.70404452782044526189E-9,
    3.85945925430276600453E-12,
    3.14040098946363334640E-15,
};
__constant__ float GD8f[] = {
    /*  1.00000000000000000000E0,*/
    1.68548898811011640017E0,
    4.87852258695304967486E-1,
    4.67913194259625806320E-2,
    1.90284426674399523638E-3,
    3.68475504442561108162E-5,
    3.57043223443740838771E-7,
    1.72693748966316146736E-9,
    3.87830166023954706752E-12,
    3.14040098946363335242E-15,
};

__constant__ double SN[] = {
    -8.39167827910303881427E-11,
    4.62591714427012837309E-8,
    -9.75759303843632795789E-6,
    9.76945438170435310816E-4,
    -4.13470316229406538752E-2,
    1.00000000000000000302E0,
};
__constant__ double SD[] = {
    2.03269266195951942049E-12,
    1.27997891179943299903E-9,
    4.41827842801218905784E-7,
    9.96412122043875552487E-5,
    1.42085239326149893930E-2,
    9.99999999999999996984E-1,
};
__constant__ double CN[] = {
    2.02524002389102268789E-11,
    -1.35249504915790756375E-8,
    3.59325051419993077021E-6,
    -4.74007206873407909465E-4,
    2.89159652607555242092E-2,
    -1.00000000000000000080E0,
};
__constant__ double CD[] = {
    4.07746040061880559506E-12,
    3.06780997581887812692E-9,
    1.23210355685883423679E-6,
    3.17442024775032769882E-4,
    5.10028056236446052392E-2,
    4.00000000000000000080E0,
};
__constant__ double FN4[] = {
    4.23612862892216586994E0,
    5.45937717161812843388E0,
    1.62083287701538329132E0,
    1.67006611831323023771E-1,
    6.81020132472518137426E-3,
    1.08936580650328664411E-4,
    5.48900223421373614008E-7,
};
__constant__ double FD4[] = {
    /*  1.00000000000000000000E0,*/
    8.16496634205391016773E0,
    7.30828822505564552187E0,
    1.86792257950184183883E0,
    1.78792052963149907262E-1,
    7.01710668322789753610E-3,
    1.10034357153915731354E-4,
    5.48900252756255700982E-7,
};
__constant__ double FN8[] = {
    4.55880873470465315206E-1,
    7.13715274100146711374E-1,
    1.60300158222319456320E-1,
    1.16064229408124407915E-2,
    3.49556442447859055605E-4,
    4.86215430826454749482E-6,
    3.20092790091004902806E-8,
    9.41779576128512936592E-11,
    9.70507110881952024631E-14,
};
__constant__ double FD8[] = {
    /*  1.00000000000000000000E0,*/
    9.17463611873684053703E-1,
    1.78685545332074536321E-1,
    1.22253594771971293032E-2,
    3.58696481881851580297E-4,
    4.92435064317881464393E-6,
    3.21956939101046018377E-8,
    9.43720590350276732376E-11,
    9.70507110881952025725E-14,
};
                                  
__constant__ double GN4[] = {
    8.71001698973114191777E-2,
    6.11379109952219284151E-1,
    3.97180296392337498885E-1,
    7.48527737628469092119E-2,
    5.38868681462177273157E-3,
    1.61999794598934024525E-4,
    1.97963874140963632189E-6,
    7.82579040744090311069E-9,
};
__constant__ double GD4[] = {
    /*  1.00000000000000000000E0,*/
    1.64402202413355338886E0,
    6.66296701268987968381E-1,
    9.88771761277688796203E-2,
    6.22396345441768420760E-3,
    1.73221081474177119497E-4,
    2.02659182086343991969E-6,
    7.82579218933534490868E-9,
};
__constant__ double GN8[] = {
    6.97359953443276214934E-1,
    3.30410979305632063225E-1,
    3.84878767649974295920E-2,
    1.71718239052347903558E-3,
    3.48941165502279436777E-5,
    3.47131167084116673800E-7,
    1.70404452782044526189E-9,
    3.85945925430276600453E-12,
    3.14040098946363334640E-15,
};
__constant__ double GD8[] = {
    /*  1.00000000000000000000E0,*/
    1.68548898811011640017E0,
    4.87852258695304967486E-1,
    4.67913194259625806320E-2,
    1.90284426674399523638E-3,
    3.68475504442561108162E-5,
    3.57043223443740838771E-7,
    1.72693748966316146736E-9,
    3.87830166023954706752E-12,
    3.14040098946363335242E-15,
};

/* Sine/cosine integrals. */
__device__ void sicif(float x, float *si, float *ci) {
    float z, c, s, f, g;
    short sign;
    
    if (x < 0.0) {
        sign = -1;
        x = -x;
    } else 
        sign = 0;

    if (x == 0.0) {
        *si = 0;
        *ci = -FLT_MAX;
        return;
    }

    if (x > 1.0e9) {
        *si = PIO2 - cosf(x)/x;
        *ci = sinf(x)/x;
    }

    if (x > 4.0) 
        goto asympt;

    z = x*x;
    s = x*polevlf(z, SNf, 5)/polevlf(z, SDf, 5);
    c = z*polevlf(z, CNf, 5)/polevlf(z, CDf, 5);

    if (sign)
        s = -s;

    *si = s;
    *ci = EUL + logf(x) + c;
    return;

asympt:
    s = sinf(x);
    c = cosf(x);
    z = 1.0/(x*x);

    if (x < 8.0) {
        f = polevlf(z, FN4f, 6)/(x*p1evlf(z, FD4f, 7));
        g = z*polevlf(z, GN4f, 7)/p1evlf(z, GD4f, 7);
    } else {
        f = polevlf(z, FN8f, 8)/(x*p1evlf(z, FD8f, 8));
        g = z*polevlf(z, GN8f, 8)/p1evlf(z, GD8f, 9);
    }
    *si = PIO2 - f*c - g*s;
    if (sign)
        *si = -(*si);
    *ci = f*s - g*c;
    return;
}

__device__ void sici(double x, double *si, double *ci) {
    double z, c, s, f, g;
    short sign;
    
    if (x < 0.0) {
        sign = -1;
        x = -x;
    } else 
        sign = 0;

    if (x == 0.0) {
        *si = 0;
        *ci = -DBL_MAX;
        return;
    }

    if (x > 1.0e9) {
        *si = PIO2 - cos(x)/x;
        *ci = sin(x)/x;
    }

    if (x > 4.0) 
        goto asympt;

    z = x*x;
    s = x*polevl(z, SN, 5)/polevl(z, SD, 5);
    c = z*polevl(z, CN, 5)/polevl(z, CD, 5);

    if (sign)
        s = -s;

    *si = s;
    *ci = EUL + log(x) + c;
    return;

asympt:
    s = sin(x);
    c = cos(x);
    z = 1.0/(x*x);

    if (x < 8.0) {
        f = polevl(z, FN4, 6)/(x*p1evl(z, FD4, 7));
        g = z*polevl(z, GN4, 7)/p1evl(z, GD4, 7);
    } else {
        f = polevl(z, FN8, 8)/(x*p1evl(z, FD8, 8));
        g = z*polevl(z, GN8, 8)/p1evl(z, GD8, 9);
    }
    *si = PIO2 - f*c - g*s;
    if (sign)
        *si = -(*si);
    *ci = f*s - g*c;
    return;
}

/* exponential integrals */
__device__ CFLOAT exp1f(CFLOAT z) {
    float x = real(z);
    float a0 = abs(z);
    CFLOAT ce1, cr, ct0, kc, ct;
    
    if (a0 == 0.0)
        ce1 = CFLOAT(1.0e300, 0.0);
    else if ((a0 < 10.0) || (x < 0.0 && a0 < 20.0)) {
        ce1 = CFLOAT(1.0, 0.0);
        cr = CFLOAT(1.0, 0.0);
        for (int k = 1; k <= 150; k++) {
            cr = -(cr * float(k) * z)/CFLOAT((k + 1.0) * (k + 1.0), 0.0);
            ce1 = ce1 + cr;
            if (abs(cr) <= abs(ce1)*1.0e-15)
                break;
        }
        ce1 = CFLOAT(-EUL,0.0)-log(z)+(z*ce1);
    } else {
        ct0 = CFLOAT(0.0, 0.0);
        for (int k = 120; k >= 1; k--) {
            kc = CFLOAT(k, 0.0);
            ct0 = kc/(CFLOAT(1.0,0.0)+(kc/(z+ct0)));
        }
        ct = CFLOAT(1.0, 0.0)/(z+ct0);
        ce1 = exp(-z)*ct;
        if (x <= 0.0 && imag(z) == 0.0)
            ce1 = ce1-CFLOAT(0.0, -PI);
    }
    return ce1;
}

__device__ CFLOAT expif(CFLOAT z) {
    CFLOAT cei = exp1f(-z);
    cei = -cei+(log(z)-log(CFLOAT(1.0)/z))/CFLOAT(2.0)-log(-z);
    return cei;
}

__device__ CDOUBLE exp1(CDOUBLE z) {
    double x = real(z);
    double a0 = abs(z);
    CDOUBLE ce1, cr, ct0, kc, ct;
    
    if (a0 == 0.0)
        ce1 = CDOUBLE(1.0e300, 0.0);
    else if ((a0 < 10.0) || (x < 0.0 && a0 < 20.0)) {
        ce1 = CDOUBLE(1.0, 0.0);
        cr = CDOUBLE(1.0, 0.0);
        for (int k = 1; k <= 150; k++) {
            cr = -(cr * double(k) * z)/CDOUBLE((k + 1.0) * (k + 1.0), 0.0);
            ce1 = ce1 + cr;
            if (abs(cr) <= abs(ce1)*1.0e-15)
                break;
        }
        ce1 = CDOUBLE(-EUL,0.0)-log(z)+(z*ce1);
    } else {
        ct0 = CDOUBLE(0.0, 0.0);
        for (int k = 120; k >= 1; k--) {
            kc = CDOUBLE(k, 0.0);
            ct0 = kc/(CDOUBLE(1.0,0.0)+(kc/(z+ct0)));
        }
        ct = CDOUBLE(1.0, 0.0)/(z+ct0);
        ce1 = exp(-z)*ct;
        if (x <= 0.0 && imag(z) == 0.0)
            ce1 = ce1-CDOUBLE(0.0, -PI);
    }
    return ce1;
}

__device__ CDOUBLE expi(CDOUBLE z) {
    CDOUBLE cei = exp1(-z);
    cei = -cei+(log(z)-log(CDOUBLE(1.0)/z))/CDOUBLE(2.0)-log(-z);
    return cei;
}

#endif /* !defined(CU_SPECIAL_FUNCS_H_) */
