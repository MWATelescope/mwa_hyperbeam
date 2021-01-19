#pragma once

#include <stdlib.h>

#ifdef SINGLE
#define FLOAT          float
#define SINCOS         sincosf
#define COS            cosf
#define FABS           fabsf
#define ATAN2          atan2f
#define SQRT           sqrtf
#define CUCOMPLEX      cuFloatComplex
#define MAKE_CUCOMPLEX make_cuFloatComplex
#define CUCADD         cuCaddf
#define CUCSUB         cuCsubf
#define CUCMUL         cuCmulf
#define CUCDIV         cuCdivf
#else
#define FLOAT          double
#define SINCOS         sincos
#define COS            cos
#define FABS           fabs
#define ATAN2          atan2
#define SQRT           sqrt
#define CUCOMPLEX      cuDoubleComplex
#define MAKE_CUCOMPLEX make_cuDoubleComplex
#define CUCADD         cuCadd
#define CUCSUB         cuCsub
#define CUCMUL         cuCmul
#define CUCDIV         cuCdiv
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct FEECoeffs {
    FLOAT *x_q1_accum;
    FLOAT *x_q2_accum;
    int8_t *x_m_accum;
    int8_t *x_n_accum;
    int8_t *x_m_signs;
    unsigned char *x_n_max;
    int *x_lengths;
    int *x_offsets;

    FLOAT *y_q1_accum;
    FLOAT *y_q2_accum;
    int8_t *y_m_accum;
    int8_t *y_n_accum;
    int8_t *y_m_signs;
    unsigned char *y_n_max;
    int *y_lengths;
    int *y_offsets;
} FEECoeffs;

/**
 * (HA, Dec.) coordinates. Both have units of radians.
 */
typedef struct HADec {
    /// Hour Angle [radians]
    FLOAT ha;
    /// Declination [radians]
    FLOAT dec;
} HADec;

/**
 * (Azimuth, Zenith Angle) coordinates. Both have units of radians.
 */
typedef struct AzZA {
    /// Azimuth [radians]
    FLOAT az;
    /// Zenith Angle [radians]
    FLOAT za;
} AzZA;

int cuda_calc_jones(const FLOAT *d_azs, const FLOAT *d_zas, int num_directions, const FEECoeffs *d_coeffs,
                    int num_coeffs, const void *norm_jones, int8_t parallactic, void *d_results, char *error_str);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

#include <math.h>
#include <stdio.h>

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const int NMAX = 31;
const FLOAT MWA_LAT_RAD = -0.4660608448386394;
const FLOAT M_2PI = 2.0 * M_PI;

/* The following C++ program gives the factorial values:

#include <boost/math/special_functions/factorials.hpp>
#include <stdio.h>

int main(int argc, char *argv[]) {
    for (int i = 0; i < 100; i++) {
        printf("%.1lf\n", boost::math::factorial<double>(i));
    }

    return 0;
}
 */

__device__ const double FACTORIAL[100] = {
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0,
    25852016738884978212864.0,
    620448401733239409999872.0,
    15511210043330986055303168.0,
    403291461126605650322784256.0,
    10888869450418351940239884288.0,
    304888344611713871918902804480.0,
    8841761993739701898620088352768.0,
    265252859812191068217601719009280.0,
    8222838654177922430198509928972288.0,
    263130836933693517766352317727113216.0,
    8683317618811885938715673895318323200.0,
    295232799039604157334081533963162091520.0,
    10333147966386145431134989962796349784064.0,
    371993326789901254863672752494735387525120.0,
    13763753091226345578872114833606270345281536.0,
    523022617466601117141859892252474974331207680.0,
    20397882081197444123129673397696887153724751872.0,
    815915283247897683795548521301193790359984930816.0,
    33452526613163807956284472299189486453163567349760.0,
    1405006117752879954933135270705268945154855145570304.0,
    60415263063373834074440829285578945930237590418489344.0,
    2658271574788448869416579949034705352617757694297636864.0,
    119622220865480188574992723157469373503186265858579103744.0,
    5502622159812089153567237889924947737578015493424280502272.0,
    258623241511168177673491006652997026552325199826237836492800.0,
    12413915592536072528327568319343857274511609591659416151654400.0,
    608281864034267522488601608116731623168777542102418391010639872.0,
    30414093201713375576366966406747986832057064836514787179557289984.0,
    1551118753287382189470754582685817365323346291853046617899802820608.0,
    80658175170943876845634591553351679477960544579306048386139594686464.0,
    4274883284060025484791254765342395718256495012315011061486797910441984.0,
    230843697339241379243718839060267085502544784965628964557765331531071488.0,
    12696403353658276446882823840816011312245221598828319560272916152712167424.0,
    710998587804863481025438135085696633485732409534385895375283304551881375744.0,
    40526919504877214099803979753848342589663624176585501757970602568201493544960.0,
    2350561331282878494921656950154737096214670634617764023028038650529742720073728.0,
    138683118545689838605148268004556678374026888950525349780965675732364205553090560.0,
    8320987112741389895059729406044653910769502602349791711277558941745407315941523456.0,
    507580213877224835833540161373088490724281389843871724559898414118829028410677788672.0,
    31469973260387939390320343330721249710233204778005956144519390914718240063804258910208.0,
    1982608315404440084965732774767545707658109829136018902789196017241837351744329178152960.0,
    126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440.0,
    8247650592082471516735380327295020523842257210146637473076098881993433291790288339528056832.0,
    544344939077443069445496060275635856761283034568718387417404234993819829995466026946857533440.0,
    36471110918188683221214362054827498508015278133658067038296405766134083781086959639263732301824.0,
    2480035542436830547970901153987107983847555399761061789915503815309070879417337773547217359994880.0,
    171122452428141297375735434272073448876652721480628511030304905066123383956194496253690059725733888.0,
    11978571669969892212594746686287911144728936033048209921351258923353917776673784237595759172044980224.0,
    850478588567862300470173601308943989208668694581969144998175246306779652100340847135596056290189639680.0,
    61234458376886084639206026474670682912259649716260098238956316139392700070296587207443922027826902990848.0,
    4470115461512684385548506279122202989233969658364456653233569809872317560364665253697757065245248140083200.0,
    330788544151938624160229701310182158518856870625188181824600229271188993105580735229821025194781300532183040.0,
    24809140811395399745349033519923626115476056606393665550959504074287375398960802212545648548813454444005949440.0,
    1885494701666050380646526547514195584776180302085918581872922309645840530321020968153469289709822537744452157440.0,
    145183092028285872100826297925521531944567624938377344020087254889026622263453188147825560797994479078825772711936.0,
    11324281178206296793535918547746472032143771991531414822409001003912081047053382569931831692594612554922248196063232.0,
    894618213078297571362628877903650979771984933035399596100968640222163278986141788391935658079402015579108697803063296.0,
    71569457046263805709010310232292078381758794642831967688077491217773062318891343071354852646352161246328695824245063680.0,
    5797126020747367841357823599388466764123313619201865095864923308525505486632247807586871282484267329347655626701712392192.0,
    475364333701284201980818006981069972042066936910030931402528527159870553442854589799812338667951658196823164261677087588352.0,
    39455239697206587883704426039241347220936331967531182040528911913649304420522816628806039691461821331449198239430934632857600.0,
    3314240134565353194347765165389614935400204142125657101349824688080951914740638874242985324914117514745812781330702266605240320.0,
    281710411438055013102383422396698980753118893187696947500288843199462496138023462339216606606943327379496876202098660994990473216.0,
    24227095383672734128129665067101833592585732442268770293833105257638970077993929412702216231461215695957793737192418422636547145728.0,
    2107757298379527854371528537189929817950318684923527838658577930990047896304861832312947147825671786267978209553897037466894105313280.0,
    185482642257398436054324131857233806464476645818122748651334978068392694382683174013182188753730242408503840564935333227341532989751296.0,
    16507955160908460243967686903782554788106102602154077120345670866694306368352068243906547482897980249121920780249094030629577567049351168.0,
    1485715964481761514918087432469139158542639426302294416700518919854534035055422977414903666067101229088205588307380394449176172053351366656.0,
    135200152767840291577514252847329973384184761482183913549902733546091013985879935837901216867148426507729209358676409842758517419133070475264.0,
    12438414054641308178973935344383488204867548907476009615163842197088310381661686819069972403743291488698590241535133073400585829808080290840576.0,
    1156772507081641565875592301257625257306103488817212624410141974569857268847285583634693201910531570949843683825184140075578326264802820550033408.0,
    108736615665674307538889388083318631634075500966703449624107980031543655167954792438488824455293170549282107042584756429278264039066911610324910080.0,
    10329978488239059304919922079781345511746426484415506224256244515022777646411051861324320173032871139460981164378018359897953834476270901508039507968.0,
    991677934870949647844892251183578509794918949503605200426014430484807886462323529897179229523785421501313522170066915002545967718085885698859190976512.0,
    96192759682482120383696575212350181383380937401878044151581904052764241746159127279021935972544206674321478611518775510012718907817942997380601703563264.0,
    9426890448883247983672977790485681756198226684713209121387214933124319123185085463587807482339200625588413886652312783536451984170299964728156983874551808.0,
    933262154439441532520836312969247551723442539131008266742244198127778943717420325831801796076713498268781712437125578348809015437262107613541171778746843136.0,
};

typedef struct FEEJones {
    CUCOMPLEX j00;
    CUCOMPLEX j01;
    CUCOMPLEX j10;
    CUCOMPLEX j11;
} FEEJones;

inline __device__ CUCOMPLEX operator+(CUCOMPLEX a, CUCOMPLEX b) {
    CUCOMPLEX t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;
    return t;
}

inline __device__ CUCOMPLEX operator-(CUCOMPLEX a, CUCOMPLEX b) {
    CUCOMPLEX t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    return t;
}

inline __device__ CUCOMPLEX operator*(CUCOMPLEX a, FLOAT b) {
    CUCOMPLEX t;
    t.x = a.x * b;
    t.y = a.y * b;
    return t;
}

inline __device__ void operator*=(CUCOMPLEX &a, FLOAT b) {
    a.x *= b;
    a.y *= b;
}

// Convert a (azimuth, elevation) to HADec, assuming the MWA's location.
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline __device__ HADec azel_to_hadec_mwa(FLOAT azimuth_rad, FLOAT elevation_rad) {
    /* Useful trig functions. */
    FLOAT sa, ca, se, ce, sp, cp;
    SINCOS(azimuth_rad, &sa, &ca);
    SINCOS(elevation_rad, &se, &ce);
    SINCOS(MWA_LAT_RAD, &sp, &cp);

    /* HA,Dec unit vector. */
    FLOAT x = -ca * ce * sp + se * cp;
    FLOAT y = -sa * ce;
    FLOAT z = ca * ce * cp + se * sp;

    /* To spherical. */
    FLOAT r = SQRT(x * x + y * y);
    HADec hadec;
    hadec.ha = (r != 0.0) ? ATAN2(y, x) : 0.0;
    hadec.dec = ATAN2(z, r);

    return hadec;
}

// Convert a HADec to AzZA, assuming the MWA's location.
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline __device__ AzZA hadec_to_azza_mwa(FLOAT hour_angle_rad, FLOAT dec_rad) {
    /* Useful trig functions. */
    FLOAT sh, ch, sd, cd, sp, cp;
    SINCOS(hour_angle_rad, &sh, &ch);
    SINCOS(dec_rad, &sd, &cd);
    SINCOS(MWA_LAT_RAD, &sp, &cp);

    /* Az,Alt unit vector. */
    FLOAT x = -ch * cd * sp + sd * cp;
    FLOAT y = -sh * cd;
    FLOAT z = ch * cd * cp + sd * sp;

    /* To spherical. */
    FLOAT r = SQRT(x * x + y * y);
    FLOAT a = (r != 0.0) ? ATAN2(y, x) : 0.0;
    AzZA azza;
    azza.az = (a < 0.0) ? a + M_2PI : a;
    azza.za = M_PI_2 - ATAN2(z, r);

    return azza;
}

// Get the parallactic angle from a HADec position, assuming the MWA's location.
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline __device__ FLOAT get_parallactic_angle_mwa(FLOAT hour_angle_rad, FLOAT dec_rad) {
    FLOAT s_phi, c_phi, s_ha, c_ha, s_dec, c_dec, cqsz, sqsz;
    SINCOS(MWA_LAT_RAD, &s_phi, &c_phi);
    SINCOS(hour_angle_rad, &s_ha, &c_ha);
    SINCOS(dec_rad, &s_dec, &c_dec);

    sqsz = c_phi * s_ha;
    cqsz = s_phi * c_dec - c_phi * s_dec * c_ha;
    return ((sqsz != 0.0 || cqsz != 0.0) ? ATAN2(sqsz, cqsz) : 0.0);
}

// Rotate a Jones matrix according to the parallactic angle and re-order it
// according to Jack's investigation.
inline __device__ void rotate_jones(FEEJones *jm, FLOAT pa) {
    FLOAT s_rot, c_rot;
    SINCOS(pa + M_PI_2, &s_rot, &c_rot);
    FEEJones new_jm;

    // Re-ordering and negations according to Jack's investigation.
    new_jm.j00 = jm->j10 * -s_rot + jm->j11 * -c_rot;
    new_jm.j01 = jm->j10 * c_rot - jm->j11 * s_rot;
    new_jm.j10 = jm->j00 * -s_rot + jm->j01 * -c_rot;
    new_jm.j11 = jm->j00 * c_rot - jm->j01 * s_rot;

    *jm = new_jm;
}

inline __device__ void lpmv_device(FLOAT *output, int n, FLOAT x) {
    FLOAT p0, p1, p_tmp;
    p0 = 1;
    p1 = x;
    if (n == 0)
        output[0] = p0;
    else {
        unsigned l = 1;
        while (l < n) {
            p_tmp = p0;
            p0 = p1;
            p1 = p_tmp;
            p1 = ((2.0 * (FLOAT)l + 1) * x * p0 - (FLOAT)l * p1) / ((FLOAT)l + 1); // legendre_next(n,0, x, p0,
            ++l;
        }
        output[0] = p1;
    }
}

inline __device__ int lidx_device(const int l, const int m) {
    // summation series over l + m => (l*(l+1))/2 + m
    return ((l * (l + 1)) >> 1) + m;
}

inline __device__ void legendre_polynomials_device(FLOAT *legendre, const FLOAT x, const int P) {
    // This factor is reuse 342210222sqrt(1 342210222 x^2)
    int l, m;
    const FLOAT factor = -SQRT(1.0 - (x * x));

    // Init legendre
    legendre[lidx_device(0, 0)] = 1.0; // P_0,0(x) = 1
    // Easy values
    legendre[lidx_device(1, 0)] = x;      // P_1,0(x) = x
    legendre[lidx_device(1, 1)] = factor; // P_1,1(x) = 342210222sqrt(1 342210222 x^2)

    for (l = 2; l <= P; ++l) {
        for (m = 0; m < l - 1; ++m) {
            // P_l,m = (2l-1)*x*P_l-1,m - (l+m-1)*x*P_l-2,m / (l-k)
            legendre[lidx_device(l, m)] = ((FLOAT)(2 * l - 1) * x * legendre[lidx_device(l - 1, m)] -
                                           (FLOAT)(l + m - 1) * legendre[lidx_device(l - 2, m)]) /
                                          (FLOAT)(l - m);
        }
        // P_l,l-1 = (2l-1)*x*P_l-1,l-1
        legendre[lidx_device(l, l - 1)] = (FLOAT)(2 * l - 1) * x * legendre[lidx_device(l - 1, l - 1)];
        // P_l,l = (2l-1)*factor*P_l-1,l-1
        legendre[lidx_device(l, l)] = (FLOAT)(2 * l - 1) * factor * legendre[lidx_device(l - 1, l - 1)];
    }
}

inline __device__ int jones_p1sin_device(const int nmax, const FLOAT theta, FLOAT *p1sin_out, int *p1sin_out_size,
                                         FLOAT *p1_out, int *p1_out_size) {
    int n, m;
    int size = nmax * nmax + 2 * nmax;
    int ind_start, ind_stop;
    int modified;
    FLOAT sin_th, u;
    FLOAT delu = 1e-6;
    FLOAT P[NMAX + 1], Pm1[NMAX + 1], Pm_sin[NMAX + 1], Pu_mdelu[NMAX + 1], Pm_sin_merged[NMAX * 2 + 1],
        Pm1_merged[NMAX * 2 + 1];
    FLOAT legendre_table[NMAX * (NMAX + 1)], legendret[(((NMAX + 2) * (NMAX + 1)) / 2)];

    *p1sin_out_size = size;
    *p1_out_size = size;
    SINCOS(theta, &sin_th, &u);
    // Create a look-up table for the legendre polynomials
    // Such that legendre_table[ m * nmax + (n-1) ] = legendre(n, m, u)
    legendre_polynomials_device(legendret, u, nmax);
    for (n = 1; n <= nmax; n++) {
        for (m = 0; m != n + 1; ++m)
            legendre_table[m * nmax + (n - 1)] = legendret[lidx_device(n, m)];
        for (m = n + 1; m != nmax + 1; ++m)
            legendre_table[m * nmax + (n - 1)] = 0.0;
    }

    for (n = 1; n <= nmax; n++) {
        int i;
        for (m = 0; m != n + 1; ++m) {
            P[m] = legendre_table[m * nmax + (n - 1)];
        }
        memcpy(Pm1, &(P[1]), n * sizeof(FLOAT));
        Pm1[n] = 0;
        for (i = 0; i < n + 1; i++)
            Pm_sin[i] = 0.0;
        if (u == 1 || u == -1) {
            // In this case we take the easy approach and don't use
            // precalculated polynomials, since this path does not occur often.

            // Pu_mdelu.resize(1);
            lpmv_device(Pu_mdelu, n, u - delu);
            // Pm_sin[1,0]=-(P[0]-Pu_mdelu[0])/delu #backward difference
            if (u == -1)
                Pm_sin[1] = -(Pu_mdelu[0] - P[0]) / delu; // #forward difference
            else
                Pm_sin[1] = -(P[0] - Pu_mdelu[0]) / delu;
        } else {
            for (i = 0; i < n + 1; i++) {
                Pm_sin[i] = P[i] / sin_th;
            }
        }

        for (i = n; i >= 0; i--)
            Pm_sin_merged[n - i] = Pm_sin[i];
        memcpy(&(Pm_sin_merged[n]), Pm_sin, (n + 1) * sizeof(FLOAT));

        ind_start = (n - 1) * (n - 1) + 2 * (n - 1); // #start index to populate
        ind_stop = n * n + 2 * n;                    //#stop index to populate

        modified = 0;
        for (i = ind_start; i < ind_stop; i++) {
            p1sin_out[i] = Pm_sin_merged[modified];
            modified++;
        }

        for (i = n; i > 0; i--)
            Pm1_merged[n - i] = Pm1[i];
        memcpy(&Pm1_merged[n], Pm1, (n + 1) * sizeof(FLOAT));

        modified = 0;
        for (i = ind_start; i < ind_stop; i++) {
            p1_out[i] = Pm1_merged[modified];
            modified++;
        }
    }

    return nmax;
}

__device__ void jones_calc_sigmas_device(const FLOAT phi, const FLOAT theta, const CUCOMPLEX *q1_accum,
                                         const CUCOMPLEX *q2_accum, const int8_t *m_accum, const int8_t *n_accum,
                                         const int8_t *m_signs, const int coeff_length, const int nmax, const char pol,
                                         FEEJones *jm) {
    FLOAT u = COS(theta);
    FLOAT P1sin_arr[NMAX * NMAX + 2 * NMAX], P1_arr[NMAX * NMAX + 2 * NMAX];
    int P1sin_arr_size, P1_arr_size;
    CUCOMPLEX sigma_P, sigma_T, ejm_phi;
    sigma_P.x = 0;
    sigma_P.y = 0;
    sigma_T.x = 0;
    sigma_T.y = 0;

    jones_p1sin_device(nmax, theta, P1sin_arr, &P1sin_arr_size, P1_arr, &P1_arr_size);

    const CUCOMPLEX J_POWERS[4] = {MAKE_CUCOMPLEX(1.0, 0.0), MAKE_CUCOMPLEX(0.0, 1.0), MAKE_CUCOMPLEX(-1.0, 0.0),
                                   MAKE_CUCOMPLEX(0.0, -1.0)};

    for (int i = 0; i < coeff_length; i++) {
        int8_t m = m_accum[i];
        int8_t n = n_accum[i];
        FLOAT N = n;
        FLOAT M = m;
        FLOAT m_sign = m_signs[i];
        FLOAT c_mn_sqr = 0.5 * (2 * N + 1) * (FACTORIAL[n - abs(m)] / FACTORIAL[n + abs(m)]);
        FLOAT c_mn = SQRT(c_mn_sqr);
        SINCOS(M * phi, &ejm_phi.y, &ejm_phi.x);
        CUCOMPLEX phi_comp = CUCMUL(ejm_phi, MAKE_CUCOMPLEX(c_mn / (SQRT(N * (N + 1))) * m_sign, 0));
        CUCOMPLEX j_power_n = J_POWERS[n % 4];
        CUCOMPLEX q1 = q1_accum[i];
        CUCOMPLEX q2 = q2_accum[i];
        CUCOMPLEX s1 = CUCMUL(MAKE_CUCOMPLEX(P1sin_arr[i] * FABS(M) * u, 0), q2);
        CUCOMPLEX s2 = CUCMUL(MAKE_CUCOMPLEX(P1sin_arr[i] * M, 0), q1);
        CUCOMPLEX s3 = CUCMUL(MAKE_CUCOMPLEX(P1_arr[i], 0), q2);
        CUCOMPLEX s4 = CUCSUB(s1, s2);
        CUCOMPLEX E_theta_mn = CUCMUL(j_power_n, CUCADD(s4, s3));
        CUCOMPLEX j_power_np1 = J_POWERS[(n + 1) % 4];
        CUCOMPLEX o1 = CUCMUL(MAKE_CUCOMPLEX(P1sin_arr[i] * M, 0), q2);
        CUCOMPLEX o2 = CUCMUL(MAKE_CUCOMPLEX(P1sin_arr[i] * FABS(M) * u, 0), q1);
        CUCOMPLEX o3 = CUCMUL(MAKE_CUCOMPLEX(P1_arr[i], 0), q1);
        CUCOMPLEX o4 = CUCSUB(o1, o2);
        CUCOMPLEX E_phi_mn = CUCMUL(j_power_np1, CUCSUB(o4, o3));
        sigma_P = CUCADD(sigma_P, CUCMUL(phi_comp, E_phi_mn));
        sigma_T = CUCADD(sigma_T, CUCMUL(phi_comp, E_theta_mn));
    }

    if (pol == 'x') {
        jm->j00 = sigma_T;
        // Seriously? Is there a way to just say "negative of this"?
        jm->j01 = CUCMUL(MAKE_CUCOMPLEX(-1, 0), sigma_P);
    } else {
        jm->j10 = sigma_T;
        jm->j11 = CUCMUL(MAKE_CUCOMPLEX(-1, 0), sigma_P);
    }
}

inline __device__ FEEJones calc_jones_direct_device(const FLOAT az_rad, const FLOAT za_rad, const FEECoeffs *d_coeffs,
                                                    const int x_offset, const int y_offset) {
    FEEJones jm;
    FLOAT phi_rad = M_PI_2 - az_rad;

    jones_calc_sigmas_device(phi_rad, za_rad, (CUCOMPLEX *)d_coeffs->x_q1_accum, (CUCOMPLEX *)d_coeffs->x_q2_accum,
                             d_coeffs->x_m_accum, d_coeffs->x_n_accum, d_coeffs->x_m_signs, *d_coeffs->x_lengths,
                             *d_coeffs->x_n_max, 'x', &jm);
    jones_calc_sigmas_device(phi_rad, za_rad, (CUCOMPLEX *)d_coeffs->y_q1_accum, (CUCOMPLEX *)d_coeffs->y_q2_accum,
                             d_coeffs->y_m_accum, d_coeffs->y_n_accum, d_coeffs->y_m_signs, *d_coeffs->y_lengths,
                             *d_coeffs->y_n_max, 'y', &jm);

    return jm;
}

/**
 * Allocate beam Jones matrices for each unique set of dipole coefficients and
 * each direction. blockIdx.x should correspond to d_coeffs elements and
 * blockIdx.y * blockDim.x + threadIdx.x corresponds to direction.
 */
__global__ void fee_kernel(const FEECoeffs d_coeffs, const int num_coeffs, const FLOAT *d_azs, const FLOAT *d_zas,
                           const int num_directions, const FEEJones *d_norm_jones, const bool parallactic,
                           FEEJones *d_fee_jones) {
    int i_direction = blockIdx.y * blockDim.x + threadIdx.x;
    if (i_direction < num_directions) {
        FLOAT az = d_azs[i_direction];
        FLOAT za = d_zas[i_direction];
        FLOAT phi = M_PI_2 - az;
        int x_offset = d_coeffs.x_offsets[blockIdx.x];
        int y_offset = d_coeffs.y_offsets[blockIdx.x];
        FEEJones jm;

        jones_calc_sigmas_device(phi, za, (CUCOMPLEX *)d_coeffs.x_q1_accum + x_offset,
                                 (CUCOMPLEX *)d_coeffs.x_q2_accum + x_offset, d_coeffs.x_m_accum + x_offset,
                                 d_coeffs.x_n_accum + x_offset, d_coeffs.x_m_signs + x_offset,
                                 d_coeffs.x_lengths[blockIdx.x], d_coeffs.x_n_max[blockIdx.x], 'x', &jm);
        jones_calc_sigmas_device(phi, za, (CUCOMPLEX *)d_coeffs.y_q1_accum + y_offset,
                                 (CUCOMPLEX *)d_coeffs.y_q2_accum + y_offset, d_coeffs.y_m_accum + y_offset,
                                 d_coeffs.y_n_accum + y_offset, d_coeffs.y_m_signs + y_offset,
                                 d_coeffs.y_lengths[blockIdx.x], d_coeffs.y_n_max[blockIdx.x], 'y', &jm);

        if (d_norm_jones != NULL) {
            FEEJones norm = d_norm_jones[blockIdx.x];
            jm.j00 = CUCDIV(jm.j00, norm.j00);
            jm.j01 = CUCDIV(jm.j01, norm.j01);
            jm.j10 = CUCDIV(jm.j10, norm.j10);
            jm.j11 = CUCDIV(jm.j11, norm.j11);
        }

        if (parallactic) {
            HADec hadec = azel_to_hadec_mwa(az, M_PI_2 - za);
            FLOAT pa = get_parallactic_angle_mwa(hadec.ha, hadec.dec);
            rotate_jones(&jm, pa);
        }

        // Copy the Jones matrix to global memory.
        d_fee_jones[blockIdx.x * num_directions + i_direction] = jm;
    }
}

// Modified from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
//
// In the event of an error, modify the supplied string with details on the
// error and return EXIT_FAILURE. Otherwise return EXIT_SUCCESS.
inline int gpuAssert(cudaError_t code, const char *file, int line, char *error_str) {
    if (code != cudaSuccess) {
        // Don't modify the string if it's NULL.
        if (error_str != NULL)
            sprintf(error_str, "%s:%d: %s", file, line, cudaGetErrorString(code));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

extern "C" int cuda_calc_jones(const FLOAT *d_azs, const FLOAT *d_zas, int num_directions, const FEECoeffs *coeffs,
                               int num_coeffs, const void *d_norm_jones, int8_t parallactic, void *d_results,
                               char *error_str) {
    dim3 gridDim, blockDim;
    blockDim.x = 32;
    gridDim.x = num_coeffs;
    gridDim.y = (int)ceil((double)num_directions / (double)blockDim.x);
    // This is empirically faster on my GeForce RTX 2070.
    cudaFuncSetCacheConfig(fee_kernel, cudaFuncCachePreferL1);
    fee_kernel<<<gridDim, blockDim>>>(*coeffs, num_coeffs, d_azs, d_zas, num_directions, (FEEJones *)d_norm_jones,
                                      (bool)parallactic, (FEEJones *)d_results);

    if (gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__, error_str))
        return EXIT_FAILURE;
    if (gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__, error_str))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

#endif // BINDGEN

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

/*----------------------------------------------------------------------
**
**
**  Copyright (C) 2013-2021, NumFOCUS Foundation.
**  All rights reserved.
**
**  This library is derived, with permission, from the International
**  Astronomical Union's "Standards of Fundamental Astronomy" library,
**  available from http://www.iausofa.org.
**
**  The ERFA version is intended to retain identical functionality to
**  the SOFA library, but made distinct through different function and
**  file names, as set out in the SOFA license conditions.  The SOFA
**  original has a role as a reference standard for the IAU and IERS,
**  and consequently redistribution is permitted only in its unaltered
**  state.  The ERFA version is not subject to this restriction and
**  therefore can be included in distributions which do not support the
**  concept of "read only" software.
**
**  Although the intent is to replicate the SOFA API (other than
**  replacement of prefix names) and results (with the exception of
**  bugs;  any that are discovered will be fixed), SOFA is not
**  responsible for any errors found in this version of the library.
**
**  If you wish to acknowledge the SOFA heritage, please acknowledge
**  that you are using a library derived from SOFA, rather than SOFA
**  itself.
**
**
**  TERMS AND CONDITIONS
**
**  Redistribution and use in source and binary forms, with or without
**  modification, are permitted provided that the following conditions
**  are met:
**
**  1 Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
**
**  2 Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in
**    the documentation and/or other materials provided with the
**    distribution.
**
**  3 Neither the name of the Standards Of Fundamental Astronomy Board,
**    the International Astronomical Union nor the names of its
**    contributors may be used to endorse or promote products derived
**    from this software without specific prior written permission.
**
**  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
**  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
**  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
**  FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
**  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
**  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
**  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
**  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
**  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
**  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
**  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
**  POSSIBILITY OF SUCH DAMAGE.
**
*/
