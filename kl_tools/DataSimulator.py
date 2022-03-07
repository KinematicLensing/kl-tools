class DataSimulator():
    ''' Base class for a grism + image data.
    
        This class works as a "data simulator": it accepts `Pars` object as 
        input arguments and generates model vector. 

        Contains astronomical images of a source at various wavelength slices.
        Records dispersion relation given specific instrument capabilities.
    '''

    #def __init__(self, data=None, shape=None, bandpasses=None, pix_scale=None,
    #             pars=None):
        ''' Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: np.array
            A numpy array containing all image slice data.
            For now, assumed to be the shape format given below.
        shape: tuple
            A 3-tuple in the format of (Nspec, Nx, Ny)
            where (Nx, Ny) are the shapes of the image slices
            and Nspec is the Number of spectral slices.
        bandpasses: list
            A list of galsim.Bandpass objects containing
            throughput function, lambda window, etc.
        pix_scale: float
            the pixel scale of the datacube slices
        pars: dict
            A dictionary that holds any additional metadata.
            In the case of grism, pars include
            {
                'diameter': diameter of telescope aperture in meter,
                'scale': pixel scale of SliceList (high-res),
                'Nx', 'Ny': grism image dimension,
                'pixel_scale': grism image pixel scale,
                'gain': detector gain,
                'exp_time': exposure time in units of second,
                'R_spec': grism resolution,
                'disp_angle': dispersion angle in units of radian,
                'offset': offset of the grism image,
                'psf_type': PSF model type, 
                'noise': noise properties
                {
                    'type': noise model type, 
                    'sky_level': sky level,
                    'read_noise': read noise,
                    'gain': detector gain, 
                }
            }
        '''
    def __init__(self, pars):
        ''' Initialize Grism data with Pars object
        
        The initialization will save the Pars object, set data dimension,
        and construct SliceList object
        Note that since we separate bandpass object into observation related
        parameters, the bandpasses during initialization is only a trivial one 
        '''
        # check and copy Pars
        utils.check_type(pars, 'pars', Pars)
        self.pars = pars

        
        # set model dimension
        self.Nx = self.pars.meta['model_dimension']['Nx']
        self.Ny = self.pars.meta['model_dimension']['Ny']
        self.scale = self.pars.meta['model_dimension']['scale']
        
        lam = self.pars.meta['model_dimension']['lambda_range']
        dlam = self.pars.meta['model_dimension']['lambda_res']
        blue_edge = np.arange(lam[0], lam[1], dlam)
        self.lambdas = np.array([(l, l+dlam) for l in blue_edge])
        self.lambda_unit = self.pars.meta['model_dimension']['lambda_unit']
        self.lambda_cen = np.mean(self.lambdas, axis=1)
        self.Nspec = self.lambdas.shape[0]
        # check convention
        self.shape = (self.Nspec, self.Ny, self.Nx)
        
        # set trivial modelcube and bandpasses
        self.bandpasses = [ gs.Bandpass(1.0, self.lambda_unit, 
            blue_limit=l[0], red_limit=l[1], 
            zeropoint=zp) for l in self.lambdas]
        # set theory model cube (trivial)
        self.modelcube = DataCube(shape=self.shape, 
            bandpasses=self.bandpasses, pix_scale=self.scale)
        
        # number of observations
        self.nobs = self.pars.meta['observations']['number_of_observations']

        return

    def evaluateTheoryModel(self, theta, rel_err=-1):
        ''' Evaluate the theory model cube based on theta and Pars
    
        theta: 1d array
            parameters being sampled

        '''
        # evaluate intensity profile, velocity distribution and SED
        self._setup_imap()
        imap_img = self.imap.render(self.pars.theta2pars(theta), 
            self.modelcube, {})
        self._setup_vmap(theta)
        X, Y = utils.build_map_grid(self.Nx, self.Ny, scale=self.scale)
        vmap_img = self.vmap('obs', X, Y, normalized=True)
        V = self.vmap('obs', X, Y)
        self._setup_SED(theta)

        # build Doppler-shifted datacube
        w_mesh = np.outer(self.lambda_cen, 1./(1.+vmap_img))
        w_mesh = w_mesh.reshape(self.lambda_cen.shape+vmap_img.shape)
        dc_array = self.sed.spectrum(w_mesh.flatten())
        dc_array = dc_array.reshape(w_mesh.shape)*imap_img[np.newaxis, :, :]
        self.modelcube.set_data(dc_array)

        return

    def getExposure(self):
        datavector = []
        covariance = []

        for obs in self.pars.meta['observations']:
            # check noise setting
            if obs.get('noise', None) is not None:
                _noise_pars = obs.get('noise', None).copy()
                _noise_pars['gain'] = obs.get('gain', 1.0)
            else:
                _noise_pars = None
            # get exposure
            if obs['type'].lower() == 'grism':
                grism_img, noise_img = self._getGrismExposure(obs, 
                    noise_pars=_noise_pars)
                datavector.append(grism_img)
                covariance.append(noise_img)
            elif obs['type'].lower() == 'photometry':
                photometry_img, noise = self._getPhotometryExposure(obs, 
                    noise_pars=_noise_pars)
                datavector.append(photometry_img)
                covariance.append(noise_img)
            else:
                raise ValueError(f'Currently only grism and photometry'+\
                    f' types observations are implemented!')
        return datavector, covariance

    def _setup_vmap(self, theta):
        ''' Setup the observer-plane LoS velocity distribution
        
        theta: list of real
            sampled parameters
        '''
        # collect related pars
        vmap_pars = self.pars.theta2pars(theta)
        vmap_pars['v_unit'] = self.pars.meta['velocity']['v_unit']
        vmap_pars['r_unit'] = self.pars.meta['velocity']['r_unit']
        model_name = self.pars.meta['velocity']['model_name'] #default/centered
        self.vmap = VelocityMap(model_name, vmap_pars)
        return

    def _setup_imap(self):
        ''' Setup the observer-plane intensity profile
        
        '''
        # collect related pars
        imap_pars = {
            'flux': 1, # SED will deal with physical flux
            'hlr': self.pars.meta['intensity']['hlr'],
            'scale': self.pars.meta['model_dimension']['scale'],
        }
        imap_type = self.pars.meta['intensity']['type'] # default/basis/inclined_exp
        self.imap = intensity.build_intensity_map(imap_type, self.datacube, 
            imap_pars)
        return

    def _setup_SED(self, theta, rel_err=-1):
        ''' Initialize the SED interpolation

        theta: 1d array
            sampled parameters. Currently the parameters being sampled
            (g1, g2, theta_int, sini, v0, vcirc, rscale) will not affect
            sed. But in general, we may consider update the SED during 
            MCMC sampling.

        '''
        self.sed = Spectrum(self.pars.meta['sed'], rel_err=rel_err)
        return


    def stack(self):
        ''' Get the dispersed 2d grism spectrum

            The function will calculate offset for each Slice caused by grism
            dispersion, and stack them to get grism spectrum

            Note: slice object structure
                    self._data = data
                    self.bandpass = bandpass
                    self.red_limit = bandpass.red_limit
                    self.blue_limit = bandpass.blue_limit
                    self.dlamda = self.red_limit - self.blue_limit
                    self.lambda_unit = bandpass.wave_type

        '''
        # apply dispersion and stack
        _grism_list = list(map(self._disperse_mapper ,self.slices))
        grism_img = np.sum(_grism_list, axis=0)
        # add noise
        noise = self._getNoise()
        if noise is not None:
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            return grism_img_withNoise, noise_img
        else:
            return grism_img, None 


    def _disperse_mapper(self, model_slice):

        area = np.pi*(self.pars['diameter']/2.)**2

        # build InterpolatedImage from model slice
        
        _img = gs.Image(model_slice.data(), make_const=True, 
                    scale=self.pars['scale'])
        _gal = gs.InterpolatedImage(_img, scale=self.pars['scale'])
        # apply PSF
        # Note that galsim.Airy takes diameter in m and lambda in nm
        PSF = self._build_PSF_model(lam=model_slice.central, 
                                    diam=self.pars['diameter']/100.)
        if PSF is not None:
            _gal = gs.Convolve([_gal, PSF])
        # calculate dispersion shift, in units of pixels
        # Note: shift = (dx[pix], dy[pix])
        shift = self._disperse(model_slice.central)
        # draw slice image
        _grism = _gal.drawImage(nx=self.pars['Nx'], ny=self.pars['Ny'], 
                                scale=self.pars['pixel_scale'], 
                                method='auto',
                                area=area, exptime=self.pars['exp_time'],
                                gain=self.pars['gain'], offset=shift,
                                bandpass=model_slice.bandpass,
                                )
        # noise will be applied after the slices are summed
        return _grism


    def _disperse(self, lam):
        ''' Get the offset due to dispersion for a given Slice

            Inputs:
                lam:    center wavelength (nm) of the slice
            Outputs:
                disp:   dispersion vector (dx, dy) in units of pixel

            This is a wrapper for instrument dispersion relation
            TODO: consider a LookUpTable for more flexible dispersion?

            For a galaxy at real position (xcen, ycen), with dispersion angle theta, the wavelength lam gets dispersed to the new position:
            
                x = xcen + (lam * dx/dlam + offset) * cos(theta),
                y = ycen + (lam * dx/dlam + offset) * sin(theta)
            
            # Note that currently the Roman grism has R = 461*lambda[um],
            # the resulting displacement in number of pixels is
            #   shift = lambda * (2/deltalambda)
            #         = lambda * (2/1um*R) = lambda / 500nm * R
            # where R is the spectral resolution at 1000nm
            # so the dispersion delta(lambda) per pixel is fixed, and
            # the displacement as a function of wavelength is a linear relation
            # But in more general cases, the dispersion relation is non-linear,
            # and interpolation is needed.
        '''
        dxdlam = self.pars['R_spec']/500. # 1/nm
        disp_direction = np.array([np.cos(self.pars['disp_ang']), 
                                   np.sin(self.pars['disp_ang'])])
        offset = self.pars['offset']
        return (lam * dxdlam + offset)*disp_direction

    def _build_PSF_model(self, **kwargs):
        psf_type = self.pars['psf_type'].lower()
        if psf_type == 'airy':
            lam = kwargs.get('lam', 1000) # nm
            diam = kwargs.get('diam', 2.4) # m
            scale_unit = kwargs.get('scale_unit', gs.arcsec)
            return gs.Airy(lam=lam, diam=diam, scale_unit=scale_unit)
        elif psf_type == 'moffat':
            beta = kwargs.get('beta', 2.5)
            fwhm = kwargs.get('fwhm', 0.5)
            return gs.Moffat(beta=beta, fwhm=fwhm)
        elif psf_type == 'none':
            return None
        else:
            raise ValueError(f'{psf_type} has not been implemented yet!')


    def _getNoise(self):
        #'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5}
        noise_pars = self.pars['noise']
        noise_type = noise_pars['type'].lower()
        random_seed = noise_pars.get('random_seed', 1314662)
        rng = gs.BaseDeviate(random_seed+1)

        if noise_type == 'ccd':
            sky_level = noise_pars.get('sky_level', 0.65*1.2)
            read_noise = noise_pars.get('read_noise', 0.85)
            gain = noise_pars.get('gain', 1.0)
            noise = gs.CCDNoise(rng=rng, gain=gain, read_noise=read_noise, 
                sky_level=sky_level)
        elif noise_type == 'gauss':
            sigma = noise_pars.get('sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif noise_type == 'poisson':
            sky_level = noise_pars.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        elif noise_type == 'none':
            noise = None
        else:
            raise ValueError(f'{noise_type} not implemented yet!')
        return noise

    def truncate(self, blue_cut, red_cut, trunc_type='edge'):
        '''
        Return a truncated DataCube to slices between blue_cut and
        red_cut using either the lambda on a slice center or edge
        '''

        for l in [blue_cut, red_cut]:
            if (not isinstance(l, float)) and (not isinstance(l, int)):
                raise ValueError('Truncation wavelengths must be ints or floats!')

        if (blue_cut >= red_cut):
            raise ValueError('blue_cut must be less than red_cut!')

        if trunc_type not in ['edge', 'center']:
            raise ValueError('trunc_type can only be at the edge or center!')

        if trunc_type == 'center':
            # truncate on slice center lambda value
            lambda_means = np.mean(self.lambdas, axis=1)

            cut = (lambda_means >= blue_cut) & (lambda_means <= red_cut)

        else:
            # truncate on slice lambda edge values
            lambda_blues = np.array([self.lambdas[i][0] for i in range(self.Nspec)])
            lambda_reds  = np.array([self.lambdas[i][1] for i in range(self.Nspec)])

            cut = (lambda_blues >= blue_cut) & (lambda_reds  <= red_cut)

        # could either update attributes or return new DataCube
        # for now, just return a new one
        trunc_data = self._data[cut,:,:]

        # Have to do it this way as lists cannot be indexed by np arrays
        # trunc_bandpasses = self.bandpasses[cut]
        trunc_bandpasses = [self.bandpasses[i]
                            for i in range(self.Nspec)
                            if cut[i] == True]

        return DataCube(data=trunc_data, bandpasses=trunc_bandpasses)


    def write(self, outfile):
        d = os.path.dirname(outfile)

        utils.make_dir(d)

        im_list = []
        for s in self.slices:
            im_list.append(s._data)

        gs.fits.writeCube(im_list, outfile)

        return