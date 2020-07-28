#ifndef _GET_FF_H__
#define _GET_FF_H__

/**
 * C++ implementation of Full Embeded Element beam model for MWA based on beam_full_EE.py
 * script and Sokolowski et al (2017) paper
 * Implemented by Marcin Sokolowski (May 2017) - marcin.sokolowski@curtin.edu.au
 */

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <mutex>

#include <boost/math/special_functions/legendre.hpp>

#include <H5Cpp.h>

#include "factorialtable.h"
#include "recursive_lock.h"

/*
  Structure for Jones matrix :
  j00 j01
  j10 j11
*/ 
class JonesMatrix 
{
public :
	std::complex<double> j00, j01, j10, j11; 

	JonesMatrix( double j00_r=0.00, double j01_r=0.00, double j10_r=0.00, double j11_r=0.00 ) : j00(j00_r,0), j01(j01_r,0), j10(j10_r,0), j11(j11_r,0)
	{ }      
	
	static void zeros(std::vector<std::vector<JonesMatrix>>& jones, size_t x_size, size_t y_size );
};

class Beam2016Implementation
{
public :
	struct DataSetIndex
	{
		DataSetIndex(char pol_, int antenna_, int freqHz_) :
			pol(pol_), antenna(antenna_), frequencyHz(freqHz_)
		{ }
		
		bool operator<(const DataSetIndex& rhs) const
		{
			return std::make_tuple(pol, antenna, frequencyHz)
				< std::make_tuple(rhs.pol, rhs.antenna, rhs.frequencyHz);
		}
		
		std::string Name() const
		{
			return pol + std::to_string(antenna+1) + '_' + std::to_string(frequencyHz);
		}
		
		char pol;
		int antenna, frequencyHz;
	};
	
	Beam2016Implementation(const double* delays, const double* amps, const std::string& searchPath );      

   //-------------------------------------------------------------------- Calculation of Jones matrix ------------------------------------------------------------------------------------------------------------------
   // Calculate jones matrix in a specified direction for a given frequency, delays and amplitudes. Zenith normalisation can also be disabled - but by default is enabled :
   // This function will  :
   //    - calculate coefficients of spherical waves (SPW) if required (meaning if frequency, delays or amplitudes are different than last set of coefficients was calculated for)
   //    - calculate electric fields (equations 4 and 5 in Sokolowski et al paper)
   //    - normalise Jones matrix to one at zenith at the same frequency (if required by parameter):
   // 
   // Only thse the functions should be used for external calls. The rest is protected from external use.
   // 
   // INPUT : (az_deg,za_deg) - azimuth and zenith angles [in degrees]
   //         freq_hz_param   - frequency in Hz
   //         delays          - delays in beamformer steps 
   //         amps            - amplitudes 
   //         bZenithNorm     - normalise to zenith (>0) or not (<=0)
   // OUTPUT : Jones matrix (normalised or not - depending on the parameter bZenithNorm )
   JonesMatrix CalcJones( double az_deg, double za_deg, int freq_hz_param, bool bZenithNorm=true );
   JonesMatrix CalcJones( double az_deg, double za_deg, int freq_hz_param, const double* delays, const double* amps, recursive_lock<std::mutex>& lock, bool bZenithNorm=true );

   // Calculation of Jones matrix for an image passed in the arrays azimuth and zenith angles maps. 
   // This function calls the single direction one (above) for all pixel in the input image.
   // INPUT : 
   //       2D array of azimuth angles in degrees
   //       2D array of zenith angles in degrees
   //       freq_hz_param   - frequency in Hz
   //       delays          - delays in beamformer steps 
   //       amps            - amplutudes          
   //       bZenithNorm     - normalise to zenith (>0) or not (<=0)
   // OUTPUT :
   //       2D array of JonesMatrix for each pixel 
   void CalcJonesArray( std::vector< std::vector<double> >& azim_arr, std::vector< std::vector<double> >& za_arr, std::vector< std::vector<JonesMatrix> >& jones,
                   int freq_hz_param, bool bZenithNorm=true );

private:
   //---------------------------------------------------- Calculations and variables for spherical waves coefficients (see equations 3-6 in the Sokolowski et al paper) ----------------------------------------------------
   // Coefficients of spherical waves (SPW) - see equations 3-6 in the Sokolowski et al paper
   // 1 refers to s=1 (transverse electric - TE modes) - eq.5 
   // 2 refers to s=2 (transverse magnetic - TM modes) - eq.5 
   // These coefficients are calculated ones for a given frequency, delays, amplitudes so these parameters are stored to 
   // only calculate new coefficients when they change.
   // X polarisation :
	 struct Coefficients
	 {
		std::vector< std::complex<double> > Q1_accum;
		std::vector< std::complex<double> > Q2_accum;
		std::vector<double> M_accum;
		std::vector<double> N_accum;
		std::vector<double> MabsM; // precalculated m/abs(m) to make it once for all pointings
		double Nmax;          // maximum N coefficient for Y (=max(N_accum_X)) - to avoid relaculations 
		std::vector<double> Cmn; // coefficient under sumation in equation 3 for X pol.
	 } _coefX, _coefY;
	 
   // Calculation of Jones matrix for a single pointing direction (internal function):
   // INPUT : (az_rad,za_rad) - azimuth and zenith in [radians]
   JonesMatrix CalcJonesDirect( double az_rad, double za_rad, const Coefficients& coefsX, const Coefficients& coefsY );
	 
   //-------------------------------------------------------------------- Calculation of Jones matrix components for given polarisation (eq. 4 and 5 in the Sokolowski et al (2017) paper --------------------------------
   // Internal function to calculate Jones matrix componenets for a given polarisation (pol). The are calculated as electric field vectors E_theta_mn (eq.4) and E_phi_mn (eq.5 in the Sokolowski et al 2017 paper).
   // INPUT :
   //    phi,theta - FEKO convention coordinates (phi=90-azim), theta=za in radians already
   //    Q1_accum, Q2_accum, M_accum, N_accum, MabsM, Nmax - coefficients calculated earlier in CalcModes function for given frequency, delays and amplitudes 
   // OUTPUT :
   //    Jones matrix filled with components for given polarisation (pol input parameter)
   void CalcSigmas(double phi, double theta, const Coefficients& coefficients, char pol, JonesMatrix& jones_matrix) const;
	
	 std::complex<double> JPower(size_t i) const
	 {
		 return _jPowerTable[i%4];
	 }
	 std::complex<double> _jPowerTable[4];
   
   // Information on last modes parameters - not to recalculate the same again and again !
   int _calcModesLastFreqHz;
   std::vector<double> _calcModesLastDelays;
   std::vector<double> _calcModesLastAmps;
   
   // function comparing current parameters : frequency, delays and amplitudes with those previously used to calculate spherical waves coefficients (stored in the 3 variables above)
   bool IsCalcModesRequired(int freq_hz, int n_ant, const double* delays, const double* amps);
   
   // Calculation of modes Q1 and Q2 and coefficients N and M and some derived variables (MabsM_X,MabsM_Y,Nmax_X and Nmax_Y) to make it once for a given pointing and 
   // then re-use for many different (azim,za) angles:
   
   // function calculating coefficients for X and Y and storing parameters frequency, delays and amplitudes 
   void GetModes(int freq_hz, size_t n_ant, const double* delays, const double* amps, Coefficients& coefsX,  Coefficients& coefsY, recursive_lock<std::mutex>& lock);
   
   // function calculating all coefficients Q1, Q2, N, M and derived MabsM, Nmax for a given polarisation ("X" or "Y") - perhaps enum should be used here 
   double CalcModes(int freq_hz, size_t n_ant, const double* delays, const double* amp, char pol, Coefficients& coefs, recursive_lock<std::mutex>& lock);

   // Calculation of normalisation matrix :
   JonesMatrix CalcZenithNormMatrix(int freq_hz, recursive_lock<std::mutex>& lock);

   std::map<int,JonesMatrix> _normJonesCache;
	 
   double _delays[16], _amps[16];

   // ----------------------------------------------------------------
   // Checking frequences included in the H5 file (stored in vector m_freq_list) :
   bool has_freq(int freq_hz) const;
   int find_closest_freq(int freq_hz) const;

	// HDF5 File interface and data structures for H5 data
	// Interface to HDF5 file format and structures to store H5 data 
	// Functions for reading H5 file and its datasets :
	// Read data from H5 file, file name is specified in the object constructor 
	void Read();
	
	const std::vector<std::vector<double>>& GetDataSet(const DataSetIndex& index, recursive_lock<std::mutex>& lock);
	
	// Read dataset_name from H5 file 
	void ReadDataSet(const std::string& dataset_name, std::vector<std::vector<double>>& out_vector, H5::H5File& h5File);
	
	// function for iteration call to H5Ovisit function :
	static herr_t list_obj_iterate(hid_t loc_id, const char *name, const H5O_info_t *info, void *operator_data);

	std::unique_ptr<H5::H5File> _h5File;
	std::string _searchPath;
	
	// Data structures for H5 file data :
	std::vector<std::string> m_obj_list;  // list of datasets in H5 file 
	std::vector<int>    m_freq_list; // list of simulated frequencies 
	std::vector< std::vector<double> > m_Modes;  // data in Modes DataSet 
	
	FactorialTable _factorial;
	
	// Calculations of Legendre polynomials :
	static void lpmv( std::vector<double>& output, int n, double x );
	
	// OUTPUT : returns list of Legendre polynomial values calculated up to order nmax :
	static int P1sin( int nmax, double theta, std::vector<double>& p1sin_out, std::vector<double>& p1_out );
	
	std::map<DataSetIndex, std::vector<std::vector<double>>> _dataSetCache;
	
	std::mutex _mutex;
};


#endif
