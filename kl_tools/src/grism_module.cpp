#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <complex>
#include <vector>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

//PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

namespace py = pybind11;

#include "galsim/GSParams.h"
#include "galsim/SBInclinedExponential.h"
#include "galsim/Image.h"
#include "galsim/Random.h"

#define idx2( i, Ni, j, Nj ) ((int)( (i) * (Nj) + (j) ))
#define idx3(i, Ni, j, Nj, k, Nk) ((int)( ( (i) * (Nj) + (j) ) * (Nk) + (k) ))
#define PI (3.14159265359)
using namespace std;

/* Parameter Structure
 * Related parameters
 * - theory cube pixel scale
 * - theory cube pixel dimension
 * - observation pixel scale
 * - observation pixel dimension
 * - bandpass file (pass as argument?)
 * - PSF (ignore chromatic PSF, pass)
 * - dispersion relation dx/dlam = R_spec / 500
 * - dispersion angle (rad)
 * - observation frame offset
 *
 * */
typedef struct {
    int theory_cube_Nx;
    int theory_cube_Ny;
    int theory_cube_Nlam;
    double theory_cube_scale;
    int observed_image_Nx;
    int observed_image_Ny;
    double observed_image_scale;
    double R_spec;
    double disper_angle;
    double offset;
} Pars;
Pars pars = {.theory_cube_Nx = 0,
             .theory_cube_Ny = 0,
             .theory_cube_Nlam = 0,
             .theory_cube_scale = 0.0,
             .observed_image_Nx = 0,
             .observed_image_Ny = 0,
             .observed_image_scale = 0.0,
             .R_spec = 0.0,
             .disper_angle = 0.0,
             .offset = 0.0};

int cpp_init_pars(int theory_cube_Nx, int theory_cube_Ny, int theory_cube_Nlam,
                   double theory_cube_scale, int observed_image_Nx,
                   int observed_image_Ny, double observed_image_scale,
                   double R_spec, double disper_angle, double offset)
{
    pars.theory_cube_Nx = theory_cube_Nx;
    pars.theory_cube_Ny = theory_cube_Ny;
    pars.theory_cube_Nlam = theory_cube_Nlam;
    pars.theory_cube_scale = theory_cube_scale;
    pars.observed_image_Nx = observed_image_Nx;
    pars.observed_image_Ny = observed_image_Ny;
    pars.observed_image_scale = observed_image_scale;
    pars.R_spec = R_spec;
    pars.disper_angle = disper_angle;
    pars.offset = offset;

    return 0;
}

int cpp_print_Pars()
{
    cout << "Print Pars object:" << endl;
    cout << "\t - theory_cube_Nx = " << pars.theory_cube_Nx << endl;
    cout << "\t - theory_cube_Ny = " << pars.theory_cube_Ny << endl;
    cout << "\t - theory_cube_Nlam = " << pars.theory_cube_Nlam << endl;
    cout << "\t - theory_cube_scale = " << pars.theory_cube_scale << endl;
    cout << "\t - observed_image_Nx = " << pars.observed_image_Nx << endl;
    cout << "\t - observed_image_Ny = " << pars.observed_image_Ny << endl;
    cout << "\t - observed_image_scale = " << pars.observed_image_scale << endl;
    cout << "\t - R_spec = " << pars.R_spec << endl;
    cout << "\t - disper_angle = " << pars.disper_angle << endl;
    cout << "\t - offset = " << pars.offset << endl;

    return 0;
}

/* Dispersion Relation
 * At the first call, the function would init the dispersion relation.
 * For a galaxy at real position (xcen,ycen), and with
 * dispersion angle theta, the wavelength lam gets dispersed
 * to the new position:
 *      x = xcen + (lam * dx/dlam + offset) * cos(theta),
 *      y = ycen + (lam * dx/dlam + offset) * sin(theta)
 * Input
 *      double lam: central wavelength in nm of the current slice
 *      vector<double> &shift: the returned resulting shift vector.
 * */
int cpp_dispersion_relation(double lam, vector<double> &shift)
{
    static int INIT = 0;
    static double dxdlam = 0.0;
    static vector<double> disp_vec{0.0, 0.0};
    // initialize dispersion relation
    if(INIT == 0){
        dxdlam = pars.R_spec / 500.0;
        disp_vec[0] = cos(pars.disper_angle);
        disp_vec[1] = sin(pars.disper_angle);
        INIT = 1;
    }
    shift[0] = (lam * dxdlam + pars.offset) * disp_vec[0];
    shift[1] = (lam * dxdlam + pars.offset) * disp_vec[1];
    return 0;
}
double obs2theory_arcsec(double center, int edge, double shift_in_pix,
                         double ref){
    return center+(edge*0.5-shift_in_pix)*pars.observed_image_scale-ref;
}
/* Generate simulated grism image out of theory 3d model cube, but with CPP
 * implementation.
 * Inputs:
 *      theory_data: the 3D model cube of dimension
 *          (Nlam_theory, Ny_theory, Nx_theory). It contains the intensity
 *          distribution $f_\lambda$
 *      lambdas: the 2D array of dimension (Nlam_theory, 2), recording the
 *          wavelength edge of each model slice.
 *      bandpasses: the 2D array of dimension (Nlam_theory, 2), recording the
 *          bandpass at the wavelength edge of each model slice.
 * Outputs:
 *      dispersed_data: the 2D image stamp of dimension (Ny, Nx) after
 *          dispersing the 3D cube to a 2D (observed) image grid.
 * */
int cpp_stack(const vector<double> &theory_data, const vector<double> &lambdas,
              const vector<double> &bandpasses, vector<double> &dispersed_data)
{
  int i,j,k;
  double l,r,t,b,lb,rb,tb,bb;
  int li,ri,ti,bi;
  // init coordinates
  // Note that those coordinates are static variables to save computation time.
  // theory model cube
  static double Rx_theory = (int)(pars.theory_cube_Nx/2) - \
                            0.5 * ((pars.theory_cube_Nx - 1) % 2);
  static double Ry_theory = (int)(pars.theory_cube_Ny/2) - \
                            0.5 * ((pars.theory_cube_Ny - 1) % 2);
  static vector<double> origin_Xgrid(pars.theory_cube_Nx, 0.0);
  static vector<double> origin_Ygrid(pars.theory_cube_Ny, 0.0);
  cout << "Rx_theory = " << Rx_theory << endl;
  cout << "Init X grid (theory cube): " << endl;
  for(i=0; i<pars.theory_cube_Nx; i++){
    origin_Xgrid[i] = (i - Rx_theory) * pars.theory_cube_scale;
    cout << origin_Xgrid[i] << " ";
  }
  cout << endl;
  cout << "Init Y grid (theory cube): " << endl;
  for(i=0; i<pars.theory_cube_Ny; i++){
    origin_Ygrid[i] = (i - Ry_theory) * pars.theory_cube_scale;
    cout << origin_Ygrid[i] << " ";
  }
  cout << endl;
  static double ob_x = origin_Xgrid[0] - 0.5*pars.theory_cube_scale;
  static double ob_y = origin_Ygrid[0] - 0.5*pars.theory_cube_scale;
  cout << "corner of the theory cube frame: "<< ob_x << ob_y << endl;
  // observed image
  static double Rx = (int)(pars.observed_image_Nx/2) - \
                      0.5 * ((pars.observed_image_Nx - 1) % 2);
  static double Ry = (int)(pars.observed_image_Ny/2) - \
                      0.5 * ((pars.observed_image_Ny - 1) % 2);
  static vector<double> target_Xgrid(pars.observed_image_Nx, 0.0);
  static vector<double> target_Ygrid(pars.observed_image_Ny, 0.0);
  cout << "Rs_obs = " << Rx << endl;
  cout << "Init X grid (observed image): " << endl;
  for(i=0; i<pars.observed_image_Nx; i++){
    target_Xgrid[i] = (i - Rx) * pars.observed_image_scale;
    cout << target_Xgrid[i] << " ";
  }
  cout << endl << "Init Y grid (observed image): " << endl;
  for(i=0; i<pars.observed_image_Ny; i++){
    target_Ygrid[i] = (i - Ry) * pars.observed_image_scale;
    cout << target_Ygrid[i] << " ";
  }
  cout << endl;

  // init dispersed_data
  for(double & it : dispersed_data){it = 0.0;}
  // looping through theory data cube
  for (i=0; i<pars.theory_cube_Nlam; i++){
    vector<double> shift{0.0, 0.0}; // in units of pixel
    double blue_limit = lambdas[2*i+0];
    double red_limit = lambdas[2*i+1];
    double mean_wave = (blue_limit + red_limit)/2.;
    double dlam = red_limit - blue_limit;
    // take the linear average of the bandpass.
    // Note that this only works when the lambda grid is fine enough.
    double mean_bp = (bandpasses[2*i+0] + bandpasses[2*i+1])/2.0;
    // for each slice, disperse & interpolate
    cpp_dispersion_relation(mean_wave, shift);
    cout << "slice " << i << " shift = (" << shift[0] << ", " << shift[1] << \
      ")" << "mean wavelength = " << mean_wave << endl;
    // loop through the dispersed image
    for(j=0; j<pars.observed_image_Ny; j++){
      for(k=0; k<pars.observed_image_Nx; k++){
        /* For each pixel in the dispersed image, find its original
         * pixels who contribute its flux. Then distribute the photons
         * from the theory cube to the observed image. If part of the
         * cell is involved, linear interpolation is applied.
         * */
        // For dispersed pixel (j,k), find its corners position in
        // arcsec, then map these corners to theory model cube, in units
        // of arcsec w.r.t. the lower-left corner of the theory model
        // cube pixel.
        l = obs2theory_arcsec(target_Xgrid[k], -1, shift[0], ob_x);
        r = obs2theory_arcsec(target_Xgrid[k], 1, shift[0], ob_x);
        b = obs2theory_arcsec(target_Ygrid[j], -1, shift[1], ob_y);
        t = obs2theory_arcsec(target_Ygrid[j], 1, shift[1], ob_y);
        lb = fmin(fmax(l/pars.theory_cube_scale, 0), pars.theory_cube_Nx);
        rb = fmin(fmax(r/pars.theory_cube_scale, 0), pars.theory_cube_Nx);
        bb = fmin(fmax(b/pars.theory_cube_scale, 0), pars.theory_cube_Ny);
        tb = fmin(fmax(t/pars.theory_cube_scale, 0), pars.theory_cube_Ny);
        li = floor(lb);
        ri = ceil(rb);
        bi = floor(bb);
        ti =  ceil(tb);
        // begin distribution
        if((li==ri) || (bi==ti)){continue;}//pixel outside the range
        else{
          int _nx = ri - li;
          int _ny = ti - bi;
          vector<double> x_weight(_nx, 1.0);
          vector<double> y_weight(_ny, 1.0);
          if(_nx > 1)
          {
            x_weight[0] = 1.0 + li - lb;
            x_weight[_nx-1] = 1.0 + rb - ri;
          }
          else{x_weight[0] = rb - lb;}

          if(_ny > 1)
          {
            y_weight[0] = 1.0 + bi - bb;
            y_weight[_ny-1] = 1.0 + tb - ti;
          }
          else{y_weight[0] = tb - bb;}
          // linear interpolation
          for(int p=0; p<_ny; p++)
          {
            for(int q=0; q<_nx; q++)
            {
              int _k = p + bi;
              int _l = q + li;
              dispersed_data[idx2(j,pars.observed_image_Ny,
                                  k,pars.observed_image_Nx)] += \
                 theory_data[idx3(i,pars.theory_cube_Nlam,\
                                  _k,pars.theory_cube_Ny,\
                                  _l,pars.theory_cube_Nx)] * \
                                  x_weight[q] * y_weight[p] * mean_bp;
            }
          }
        }
        // end distribution
      }
    }
  }
  return 0;
}

/* PYBIND11 Python Wrapper
 * */
PYBIND11_MODULE(kltools_grism_module, m) {

  py::bind_vector<std::vector<double>>(m, "DBVec");

  m.doc() = "cpp grism module"; // optional module docstring

  m.def("init_pars", &cpp_init_pars, "A function that init Pars struct");

  m.def("print_Pars", &cpp_print_Pars, "A function that print Pars struct");

  m.def("stack", &cpp_stack,
        "A function that disperse and stack the theory model cube");

  #ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
  m.attr("__version__") = "dev";
  #endif
}


