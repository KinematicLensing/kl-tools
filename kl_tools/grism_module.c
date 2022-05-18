#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define idx( i, Ni, j, Nj ) ((int)( (i) * (Nj) + (j) ))

void disperse(double *theory_slice, double *shift, 
	float scale, int Nx_theory, int Ny_theory, 
	float pix_scale, int Nx, int Ny, 
	double *target_img, 
	float *target_Xmesh, float *target_Ymesh, 
	float *origin_Xmesh, float* origin_Ymesh)
{
	double shift_arcsec[4] = {
		shift[0] * scale,
		shift[0] * scale,
		shift[1] * scale,
		shift[1] * scale,
	};
	int i,j,k,l;
	double _target_arcsec[4]; // x left, x right, y bottom, y top
	double _origin_arcsec[4]; // x left, x right, y bottom, y top
	double _origin_pixel[4];  // x left, x right, y bottom, y top
	double _pixel_bound[4];   // x left, x right, y bottom, y top
	double _origin_corner[4] = {
		origin_Xmesh[0] - 0.5 * scale,
		origin_Xmesh[0] - 0.5 * scale,
		origin_Ymesh[0] - 0.5 * scale,
		origin_Ymesh[0] - 0.5 * scale,
	};
	double *x_weight, *y_weight;
	int _pixel_impact[4];     // x left, x right, y bottom, y top
	
	for(i=0; i<Ny; i++)
	{
		for(j=0; j<Nx; j++)
		{
			// target pixel in observed detector, [arcsec]
			// origin: center
			_target_arcsec[0] = target_Xmesh[j] - 0.5*pix_scale;
			_target_arcsec[1] = target_Xmesh[j] + 0.5*pix_scale;
			_target_arcsec[2] = target_Ymesh[i] - 0.5*pix_scale;
			_target_arcsec[3] = target_Ymesh[i] + 0.5*pix_scale;
			// target pixel -> theory model "origin pixel", [arcsec]
			// origin: lower-left
			for(k=0; k<4; k++)
			{
				_origin_arcsec[k] = _target_arcsec[k] - shift_arcsec[k] - _origin_corner[k];
				// origin pixel [arcsec] -> [pixel]
				_origin_pixel[k] = _origin_arcsec[k] / scale;
			}
			// bounded by the theory model cube boundary
			_pixel_bound[0] = fmin(fmax(_origin_pixel[0], 0), Nx_theory);
			_pixel_bound[1] = fmin(fmax(_origin_pixel[1], 0), Nx_theory);
			_pixel_bound[2] = fmin(fmax(_origin_pixel[2], 0), Ny_theory);
			_pixel_bound[3] = fmin(fmax(_origin_pixel[3], 0), Ny_theory);
			// origin: lower-left, starting from 0, [start, end)
			_pixel_impact[0] = floor(_pixel_bound[0]);
			_pixel_impact[1] =  ceil(_pixel_bound[1]);
			_pixel_impact[2] = floor(_pixel_bound[2]);
			_pixel_impact[3] =  ceil(_pixel_bound[3]);

			if((_pixel_impact[0]==_pixel_impact[1]) || (_pixel_impact[2]==_pixel_impact[3]))
			{
				continue;
			}
			else
			{
				int _nx = _pixel_impact[1]-_pixel_impact[0];
				int _ny = _pixel_impact[3]-_pixel_impact[2];
				x_weight = (double *) calloc(_nx, sizeof(double));
				y_weight = (double *) calloc(_ny, sizeof(double));
				if(_nx > 1)
				{
					x_weight[0] = 1.0 + _pixel_impact[0] - _pixel_bound[0];
					x_weight[_nx-1] = 1.0 + _pixel_bound[1] - _pixel_impact[1];
					for(k=1; k<_nx-1; k++){x_weight[k] = 1.0;}
				}
				else
				{
					x_weight[0] = _pixel_bound[1] - _pixel_bound[0];
				}
				if(_ny > 1)
				{
					y_weight[0] = 1.0 + _pixel_impact[2] - _pixel_bound[2];
					y_weight[_ny-1] = 1.0 + _pixel_bound[3] - _pixel_impact[3];
					for(k=1; k<_ny-1; k++){y_weight[k] = 1.0;}
				}
				else
				{
					y_weight[0] = _pixel_bound[3] - _pixel_bound[2];
				}
				for(k=0; k<_ny; k++)
				{
					for(l=0; l<_nx; l++)
					{
						int _k = k + _pixel_impact[2];
						int _l = l + _pixel_impact[0];
						target_img[idx(i,Ny,j,Nx)] += theory_slice[idx(_k,Ny_theory,_l,Nx_theory)] * x_weight[l] * y_weight[k];
					}
				}
				free(x_weight);
				free(y_weight);
			}
			
		}
	}
}

int main()
{
	int i,j,k,l;
	// theory model cube
	float scale = 0.05;
	int Nx_theory = 60, Ny_theory=60;
	float Rx_theory = (int)(Nx_theory/2) - 0.5 * ((Nx_theory - 1) % 2);
	float Ry_theory = (int)(Ny_theory/2) - 0.5 * ((Ny_theory - 1) % 2);
	float *origin_Xgrid = (float *)calloc(Nx_theory, sizeof(float));
	float *origin_Ygrid = (float *)calloc(Ny_theory, sizeof(float));
	for(i=0; i<Nx_theory; i++){
		origin_Xgrid[i] = (Rx_theory + i) * scale;
	}
	for(i=0; i<Ny_theory; i++){
		origin_Ygrid[i] = (Ry_theory + i) * scale;
	}

	// detector pixel 
	float pix_scale = 0.13;
	int Nx = 36, Ny=36;
	float Rx = (int)(Nx/2) - 0.5 * ((Nx - 1) % 2);
	float Ry = (int)(Ny/2) - 0.5 * ((Ny - 1) % 2);
	float *target_Xgrid = (float *)calloc(Nx, sizeof(float));
	float *target_Ygrid = (float *)calloc(Ny, sizeof(float));
	for(i=0; i<Nx; i++){
		target_Xgrid[i] = (Rx + i) * pix_scale; 
	}
	for(i=0; i<Ny; i++){
		target_Ygrid[i] = (Ry + i) * pix_scale;
	}
	double *target_img = (double *)calloc(Nx * Ny, sizeof(double));
	
	// lambda grid
	float lambda_blue = 1260.0, lambda_red = 1302.0;
	float lambda_res = 0.5; // nm
	int Nlam = (int)((lambda_red - lambda_blue)/lambda_res);
	double *lambdas = (double *)calloc(Nlam, sizeof(double));

	// grism related
	double disp_ang = 0.0; // rad, 0, 1.57
	double dxdlam = 4.65; // nm / pixel
	double offset = -275.48161224045805; // pixel
	double *shift = (double *)calloc(2*Nlam, sizeof(double));

	for(i=0; i<Nlam; i++)
	{
		lambdas[i] = lambda_blue + (i+0.5)*lambda_res;
		shift[i*2]   = (lambdas[i] * dxdlam + offset) * cos(disp_ang);
		shift[i*2+1] = (lambdas[i] * dxdlam + offset) * sin(disp_ang);
	}

	// theory cube
	double *theory_slice = (double *)calloc(Nlam * Ny_theory * Nx_theory, sizeof(double));

	printf("Theory model cube size (%d, %d, %d)\n", Nx_theory, Ny_theory, Nlam);
	// run dispersion
	clock_t start = clock();
	for(i=0; i<Nlam; i++)
	{
		double *_slice = theory_slice + i * Nx_theory * Ny_theory;
		double *_shift = shift + i * 2;
		disperse(_slice, _shift, 
			scale, Nx_theory, Ny_theory, 
			pix_scale, Nx, Ny, target_img, 
			target_Xgrid, target_Ygrid, 
			origin_Xgrid, origin_Ygrid);
	}
	clock_t stop = clock();
	double time_spent = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Done (%f ms)\n", time_spent*1000.);
}