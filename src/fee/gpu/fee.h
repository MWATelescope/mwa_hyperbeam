#pragma once

#include "gpu_common.cuh"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct FEECoeffs {
    const FLOAT *x_q1_accum;
    const FLOAT *x_q2_accum;
    const int8_t *x_m_accum;
    const int8_t *x_n_accum;
    const int8_t *x_m_signs;
    const int8_t *x_m_abs_m;
    const int *x_lengths;
    const int *x_offsets;

    const FLOAT *y_q1_accum;
    const FLOAT *y_q2_accum;
    const int8_t *y_m_accum;
    const int8_t *y_n_accum;
    const int8_t *y_m_signs;
    const int8_t *y_m_abs_m;
    const int *y_lengths;
    const int *y_offsets;

    const unsigned char n_max;
} FEECoeffs;

const char *gpu_calc_jones(const FLOAT *d_azs, const FLOAT *d_zas, int num_directions, const FEECoeffs *d_coeffs,
                           int num_coeffs, const void *d_norm_jones, const FLOAT *d_latitude_rad,
                           const int iau_reorder, void *d_results);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const int NMAX = 31;

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

// Apply the parallactic-angle correction. If `iau_order` is true then the beam
// response is [NS-NS NS-EW EW-NS EW-EW], otherwise [EW-EW EW-NS NS-EW NS-NS].
inline __device__ void apply_pa_correction(JONES *jm, FLOAT pa, int iau_order) {
    FLOAT s_rot, c_rot;
    SINCOS(pa, &s_rot, &c_rot);

    if (iau_order == 1) {
        *jm = JONES{
            .j00 = CADD(jm->j10 * -c_rot, jm->j11 * s_rot),
            .j01 = CADD(jm->j10 * -s_rot, jm->j11 * -c_rot),
            .j10 = CADD(jm->j00 * -c_rot, jm->j01 * s_rot),
            .j11 = CADD(jm->j00 * -s_rot, jm->j01 * -c_rot),
        };
    } else {
        *jm = JONES{
            .j00 = CADD(jm->j00 * -s_rot, jm->j01 * -c_rot),
            .j01 = CADD(jm->j00 * -c_rot, jm->j01 * s_rot),
            .j10 = CADD(jm->j10 * -s_rot, jm->j11 * -c_rot),
            .j11 = CADD(jm->j10 * -c_rot, jm->j11 * s_rot),
        };
    }
}

inline __device__ void lpmv_device(FLOAT *output, int n, FLOAT x) {
    FLOAT p0, p1, p_tmp;
    p0 = 1;
    p1 = x;
    if (n == 0)
        output[0] = p0;
    else {
        int l = 1;
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

inline __device__ int jones_p1sin_device(const int nmax, const FLOAT theta, FLOAT *p1sin_out, FLOAT *p1_out) {
    int n, m;
    int ind_start, ind_stop;
    int modified;
    FLOAT sin_th, u;
    const FLOAT delu = 1e-6;
    FLOAT P[NMAX + 1], Pm1[NMAX + 1], Pm_sin[NMAX + 1], Pu_mdelu[NMAX + 1], Pm_sin_merged[NMAX * 2 + 1],
        Pm1_merged[NMAX * 2 + 1];
    FLOAT legendre_table[NMAX * (NMAX + 1)], legendret[(((NMAX + 2) * (NMAX + 1)) / 2)];

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
        ind_stop = n * n + 2 * n;                    // #stop index to populate

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

inline __device__ void jones_calc_sigmas_device(const FLOAT phi, const FLOAT theta, const COMPLEX *q1_accum,
                                                const COMPLEX *q2_accum, const int8_t *m_accum, const int8_t *n_accum,
                                                const int8_t *m_signs, const int8_t *m_abs_m, const int coeff_length,
                                                const FLOAT *P1sin_arr, const FLOAT *P1_arr, const char pol,
                                                JONES *jm) {
    const FLOAT u = COS(theta);
    COMPLEX sigma_P = MAKE_COMPLEX(0.0, 0.0);
    COMPLEX sigma_T = MAKE_COMPLEX(0.0, 0.0);
    COMPLEX ejm_phi;

    const COMPLEX J_POWERS[4] = {MAKE_COMPLEX(1.0, 0.0), MAKE_COMPLEX(0.0, 1.0), MAKE_COMPLEX(-1.0, 0.0),
                                 MAKE_COMPLEX(0.0, -1.0)};

    for (int i = 0; i < coeff_length; i++) {
        const int8_t m = m_accum[i];
        const int8_t n = n_accum[i];
        const FLOAT N = n;
        const FLOAT M = m;
        const FLOAT m_sign = m_signs[i];
        const int8_t m_abs = m_abs_m[i];
        const FLOAT c_mn_sqr = 0.5 * (2 * N + 1) * (FACTORIAL[n - m_abs] / FACTORIAL[n + m_abs]);
        const FLOAT c_mn = SQRT(c_mn_sqr);
        SINCOS(M * phi, &ejm_phi.y, &ejm_phi.x);
        const COMPLEX phi_comp = ejm_phi * (c_mn / (SQRT(N * (N + 1))) * m_sign);
        const COMPLEX j_power_n = J_POWERS[n % 4];
        const COMPLEX q1 = q1_accum[i];
        const COMPLEX q2 = q2_accum[i];
        const COMPLEX s1 = q2 * (P1sin_arr[i] * FABS(M) * u);
        const COMPLEX s2 = q1 * (P1sin_arr[i] * M);
        const COMPLEX s3 = q2 * P1_arr[i];
        const COMPLEX s4 = CSUB(s1, s2);
        const COMPLEX E_theta_mn = CMUL(j_power_n, CADD(s4, s3));
        const COMPLEX j_power_np1 = J_POWERS[(n + 1) % 4];
        const COMPLEX o1 = q2 * (P1sin_arr[i] * M);
        const COMPLEX o2 = q1 * (P1sin_arr[i] * FABS(M) * u);
        const COMPLEX o3 = q1 * P1_arr[i];
        const COMPLEX o4 = CSUB(o1, o2);
        const COMPLEX E_phi_mn = CMUL(j_power_np1, CSUB(o4, o3));
        sigma_P += CMUL(phi_comp, E_phi_mn);
        sigma_T += CMUL(phi_comp, E_theta_mn);
    }

    if (pol == 'x') {
        jm->j00 = sigma_T;
        jm->j01 = sigma_P * -1.0;
    } else {
        jm->j10 = sigma_T;
        jm->j11 = sigma_P * -1.0;
    }
}

/**
 * Allocate beam Jones matrices for each unique set of dipole coefficients and
 * each direction. blockIdx.y should correspond to d_coeffs elements and
 * blockIdx.x * blockDim.x + threadIdx.x corresponds to direction.
 */
__global__ void fee_kernel(const FEECoeffs coeffs, const FLOAT *azs, const FLOAT *zas, const int num_directions,
                           const JONES *norm_jones, const FLOAT *latitude_rad, const int iau_order,
                           JONES *fee_jones) {
    for (int i_direction = blockIdx.x * blockDim.x + threadIdx.x; i_direction < num_directions;
         i_direction += gridDim.x * blockDim.x) {
        const FLOAT az = azs[i_direction];
        const FLOAT za = zas[i_direction];
        const FLOAT phi = M_PI_2 - az;

        // Set up our "P1sin" arrays. This is pretty expensive, but only depends
        // on the zenith angle and "n_max".
        FLOAT P1sin_arr[NMAX * NMAX + 2 * NMAX], P1_arr[NMAX * NMAX + 2 * NMAX];
        jones_p1sin_device(coeffs.n_max, za, P1sin_arr, P1_arr);

        const int x_offset = coeffs.x_offsets[blockIdx.y];
        const int y_offset = coeffs.y_offsets[blockIdx.y];
        JONES jm;
        jones_calc_sigmas_device(phi, za, (const COMPLEX *)coeffs.x_q1_accum + x_offset,
                                 (const COMPLEX *)coeffs.x_q2_accum + x_offset, coeffs.x_m_accum + x_offset,
                                 coeffs.x_n_accum + x_offset, coeffs.x_m_signs + x_offset, coeffs.x_m_abs_m + x_offset,
                                 coeffs.x_lengths[blockIdx.y], P1sin_arr, P1_arr, 'x', &jm);
        jones_calc_sigmas_device(phi, za, (const COMPLEX *)coeffs.y_q1_accum + y_offset,
                                 (const COMPLEX *)coeffs.y_q2_accum + y_offset, coeffs.y_m_accum + y_offset,
                                 coeffs.y_n_accum + y_offset, coeffs.y_m_signs + y_offset, coeffs.y_m_abs_m + y_offset,
                                 coeffs.y_lengths[blockIdx.y], P1sin_arr, P1_arr, 'y', &jm);

        if (norm_jones != NULL) {
            JONES norm = norm_jones[blockIdx.y];
            jm.j00 = CDIV(jm.j00, norm.j00);
            jm.j01 = CDIV(jm.j01, norm.j01);
            jm.j10 = CDIV(jm.j10, norm.j10);
            jm.j11 = CDIV(jm.j11, norm.j11);
        }

        if (latitude_rad != NULL) {
            HADec hadec = azel_to_hadec(az, M_PI_2 - za, *latitude_rad);
            FLOAT pa = get_parallactic_angle(hadec, *latitude_rad);
            apply_pa_correction(&jm, pa, iau_order);
        }

        // Copy the Jones matrix to global memory.
        fee_jones[blockIdx.y * num_directions + i_direction] = jm;
    }
}

extern "C" const char *gpu_calc_jones(const FLOAT *d_azs, const FLOAT *d_zas, int num_directions,
                                      const FEECoeffs *d_coeffs, int num_coeffs, const void *d_norm_jones,
                                      const FLOAT *d_latitude_rad, const int iau_order, void *d_results) {
    dim3 gridDim, blockDim;
    blockDim.x = warpSize;
    gridDim.x = (int)ceil((double)num_directions / (double)blockDim.x);
    gridDim.y = num_coeffs;
    fee_kernel<<<gridDim, blockDim>>>(*d_coeffs, d_azs, d_zas, num_directions, (JONES *)d_norm_jones,
                                      d_latitude_rad, iau_order, (JONES *)d_results);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}

#endif // BINDGEN

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
