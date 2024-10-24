import numpy as np
import healpy as hp  # HEALPix for handling spherical maps
from astropy.table import Table
from pathlib import Path
from sklearn.neighbors import KernelDensity  # For density estimation
import wget  # For downloading files
from tqdm import tqdm  # For progress bars


class LensingMocks:
    """
    A class for handling lensing covariance computations.
    """

    def __init__(self, download_dir, output_dir, nres, nsbins):
        """
        Initializes directories, resolution, and number of bins.

        Parameters:
        - download_dir (str): Directory to download data
        - output_dir (str): Directory to output data
        - nres (int): Resolution parameter (power of 2)
        - nsbins (int): Number of source bins
        """
        self.download_dir = Path(download_dir)
        self.output_dir = Path(output_dir)
        self.nres = nres
        self.nside = 2 ** self.nres
        self.npix = hp.nside2npix(nside=self.nside)
        self.nsbins = nsbins

    def download_file(self, url, filename, overwrite):
        """
        Downloads a file from a specified URL.

        Parameters:
        - url (str): URL of the file to download
        - filename (Path): Local filename to save the file
        - overwrite (bool): If True, overwrite existing files
        """
        if filename.exists() and not overwrite:
            print(f"{filename} already exists")
        else:
            wget.download(url, out=str(filename), bar=wget.bar_adaptive)
            print(f"{filename} downloaded")

    def download_lensing_files(self, zindex, los, overwrite=False):
        """
        Downloads lensing files from the specified URL.

        Parameters:
        - zindex (int): Redshift index
        - los (int): Line of sight index
        - overwrite (bool): If True, overwrite existing files
        """
        subdir = 'sub1' if los < 54 else 'sub2'
        url = (f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/{subdir}/nres{self.nres}/'
               f'allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat')
        filename = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat'
        self.download_file(url, filename, overwrite)

    def download_density_files(self, los, overwrite=False):
        """
        Downloads density files from the specified URL.

        Parameters:
        - los (int): Line of sight index
        - overwrite (bool): If True, overwrite existing files
        """
        subdir = 'sub1' if los < 54 else 'sub2'
        url = (f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/{subdir}/nres{self.nres}/'
               f'delta_shell_maps/allskymap_nres{self.nres}r{los:03}.delta_shell.dat')
        filename = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
        self.download_file(url, filename, overwrite)

    def check_files(self, los):
        """
        Checks if all necessary lensing and density files are available for the given line of sight.
        If not, it attempts to download them.

        Parameters:
        - los (int): Line of sight index
        """
        all_files_available = True
        for zbin in range(1, self.nzbins + 1):
            filename = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.zs{zbin}.mag.dat'
            if not filename.exists():
                print(f'Downloading lensing file {filename}')
                self.download_lensing_files(zindex=zbin, los=los, overwrite=True)
                all_files_available = False

        if all_files_available:
            print('All source files available')

        density_filename = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
        if not density_filename.exists():
            print(f'Downloading density file {density_filename}')
            self.download_density_files(los=los, overwrite=True)
        else:
            print(f'Density file {density_filename} available')

    def load_maps(self, z, los):
        """
        Loads the lensing maps (kappa, gamma1, gamma2) for a given redshift bin and line of sight.

        Parameters:
        - z (int): Redshift bin index
        - los (int): Line of sight index

        Returns:
        - kappa (np.ndarray): Convergence map
        - gamma1 (np.ndarray): Shear component gamma1
        - gamma2 (np.ndarray): Shear component gamma2
        """
        filename = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.zs{z}.mag.dat'

        # Skip indices for reading binary data
        skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
        load_blocks = [skip[i + 1] - skip[i] for i in range(len(skip) - 1)]

        with open(filename, 'rb') as f:
            # Read headers
            rec = np.fromfile(f, dtype='uint32', count=1)
            nside = np.fromfile(f, dtype='int32', count=1)[0]
            npix = np.fromfile(f, dtype='int64', count=1)[0]
            self.npix = npix  # Update npix
            np.fromfile(f, dtype='uint32', count=2)  # Skip record markers

            def read_component():
                data = np.empty(npix, dtype='float32')
                idx = 0
                r = npix
                for l in load_blocks:
                    blocks = min(l, r)
                    if blocks <= 0:
                        break
                    data[idx:idx + blocks] = np.fromfile(f, dtype='float32', count=blocks)
                    np.fromfile(f, dtype='uint32', count=2)  # Skip record markers
                    idx += blocks
                    r -= blocks
                return data

            # Read components
            kappa = read_component()
            gamma1 = read_component()
            gamma2 = read_component()

        return kappa, gamma1, gamma2

    @staticmethod
    def rotate_gals(ras, decs, gammas1, gammas2, rotangle, inv=False, units="deg"):
        """
        Rotates survey patch so that its center of mass lies at the origin.

        Parameters:
        - ras (np.ndarray): Right Ascension array
        - decs (np.ndarray): Declination array
        - gammas1 (np.ndarray): Shear component gamma1
        - gammas2 (np.ndarray): Shear component gamma2
        - rotangle (float): Rotation angle
        - inv (bool): Whether to perform inverse rotation
        - units (str): Units of input angles ("deg" or "rad")

        Returns:
        - ra_rot (np.ndarray): Rotated Right Ascension
        - dec_rot (np.ndarray): Rotated Declination
        - gamma1_rot (np.ndarray): Rotated shear component gamma1
        - gamma2_rot (np.ndarray): Rotated shear component gamma2
        """
        # Convert to radians if necessary
        if units == "deg":
            decs_rad = np.deg2rad(decs)
            ras_rad = np.deg2rad(ras)
            rotangle_rad = np.deg2rad(rotangle)
        else:
            decs_rad = decs
            ras_rad = ras
            rotangle_rad = rotangle

        thetas = np.pi / 2 - decs_rad
        phis = ras_rad

        # Compute rotation
        rotator = hp.Rotator(rot=[0, rotangle_rad], deg=False, inv=inv)
        rotated_thetas, rotated_phis = rotator(thetas, phis)

        # Rotate shear components
        angle = rotator.angle_ref(rotated_thetas, rotated_phis, inv=True)
        gamma_rot = (gammas1 + 1j * gammas2) * np.exp(2j * angle)

        # Transform back to (ra, dec)
        ra_rot = rotated_phis
        dec_rot = np.pi / 2 - rotated_thetas
        if units == "deg":
            dec_rot = np.rad2deg(dec_rot)
            ra_rot = np.rad2deg(ra_rot)

        return ra_rot, dec_rot, gamma_rot.real, gamma_rot.imag

    def create_lensing_maps(self, los):
        """
        Creates lensing maps (gamma1, gamma2, kappa) for all source bins by summing over redshift bins.

        Parameters:
        - los (int): Line of sight index
        """
        self.npix = hp.nside2npix(self.nside)
        kappa_allbins = np.zeros((self.nsbins, self.npix), dtype=np.float32)
        gamma1_allbins = np.zeros((self.nsbins, self.npix), dtype=np.float32)
        gamma2_allbins = np.zeros((self.nsbins, self.npix), dtype=np.float32)

        for zbin in tqdm(range(self.nzbins), desc="Loading redshift bins"):
            kappa_zbin, gamma1_zbin, gamma2_zbin = self.load_maps(z=zbin + 1, los=los)
            if np.any(np.isnan([kappa_zbin, gamma1_zbin, gamma2_zbin])):
                print(f'Lens maps have NaNs in redshift bin {zbin + 1}')

            weights = self.Nz_persbin_weights[:, zbin][:, np.newaxis]
            kappa_allbins += weights * kappa_zbin
            gamma1_allbins += weights * gamma1_zbin
            gamma2_allbins += weights * gamma2_zbin

        for sbin in range(self.nsbins):
            hp.write_map(self.output_dir / f'gamma1_nside{self.nside}_tomobin{sbin + 1}_{los}.fits',
                         gamma1_allbins[sbin], dtype=np.float32, overwrite=True)
            hp.write_map(self.output_dir / f'gamma2_nside{self.nside}_tomobin{sbin + 1}_{los}.fits',
                         gamma2_allbins[sbin], dtype=np.float32, overwrite=True)
            hp.write_map(self.output_dir / f'kappa_nside{self.nside}_tomobin{sbin + 1}_{los}.fits',
                         kappa_allbins[sbin], dtype=np.float32, overwrite=True)

        print(f'Finished writing shear maps for LOS {los}')

    def loading_shear_maps(self, sbin, los):
        """
        Loads the shear and convergence maps for a given source bin and line of sight.

        Parameters:
        - sbin (int): Source bin index
        - los (int): Line of sight index

        Returns:
        - gamma1 (np.ndarray): Shear component gamma1
        - gamma2 (np.ndarray): Shear component gamma2
        - kappa (np.ndarray): Convergence map
        """
        gamma1 = hp.read_map(self.output_dir / f'gamma1_nside{self.nside}_tomobin{sbin}_{los}.fits')
        gamma2 = hp.read_map(self.output_dir / f'gamma2_nside{self.nside}_tomobin{sbin}_{los}.fits')
        kappa = hp.read_map(self.output_dir / f'kappa_nside{self.nside}_tomobin{sbin}_{los}.fits')

        print(f'Finished loading shear maps for LOS {los}, bin {sbin}')

        return gamma1, gamma2, kappa

    @staticmethod
    def add_noise_obs(g1, g2, eps1, eps2):
        """
        Adds observational noise to the reduced shear, incorporating intrinsic ellipticity.

        Parameters:
        - g1 (np.ndarray): Reduced shear component g1
        - g2 (np.ndarray): Reduced shear component g2
        - eps1 (np.ndarray): Intrinsic ellipticity component e1
        - eps2 (np.ndarray): Intrinsic ellipticity component e2

        Returns:
        - e1_obs (np.ndarray): Observed ellipticity component e1
        - e2_obs (np.ndarray): Observed ellipticity component e2
        """
        g = g1 + 1j * g2
        mask = np.abs(g) < 1

        eps_int = (eps1 + 1j * eps2) * np.exp(-2j * np.random.uniform(0, 2 * np.pi, size=len(eps1)))
        epsilon_obs = np.empty_like(eps_int)

        # For |g| < 1
        epsilon_obs[mask] = (g[mask] + eps_int[mask]) / (1 + np.conj(g[mask]) * eps_int[mask])

        # For |g| >= 1
        epsilon_obs[~mask] = (1 + g[~mask] * np.conj(eps_int[~mask])) / (np.conj(g[~mask]) + np.conj(eps_int[~mask]))

        return epsilon_obs.real, epsilon_obs.imag

    def create_real_shear_catalogue(self, sbin, los, ra_sources, dec_sources, e_1_rot, e_2_rot, weights):
        """
        Creates a shear catalogue by applying shear to the source galaxies and adding noise.

        Parameters:
        - sbin (int): Source bin index
        - los (int): Line of sight index
        - ra_sources (np.ndarray): Right Ascension of sources
        - dec_sources (np.ndarray): Declination of sources
        - e_1_rot (np.ndarray): Rotated intrinsic ellipticity component e1
        - e_2_rot (np.ndarray): Rotated intrinsic ellipticity component e2
        - weights (np.ndarray): Weights for the sources

        Returns:
        - gamma_table (Table): Astropy Table containing shear catalogue
        """
        # Load shear maps
        gamma1_map, gamma2_map, kappa_map = self.loading_shear_maps(sbin, los)

        # Find HEALPix pixels corresponding to source positions
        pix_indices = hp.ang2pix(nside=self.nside, theta=ra_sources, phi=dec_sources, lonlat=True)

        # Get shear and convergence values at source positions
        kappa_values = kappa_map[pix_indices]
        gamma1_values = gamma1_map[pix_indices]
        gamma2_values = gamma2_map[pix_indices]

        # Compute reduced shear
        g1 = gamma1_values / (1 - kappa_values)
        g2 = gamma2_values / (1 - kappa_values)

        # Add observational noise
        e1_obs, e2_obs = self.add_noise_obs(g1, g2, e_1_rot, e_2_rot)

        # Create an Astropy Table to store the data
        gamma_table = Table({
            'ra': ra_sources.astype(np.float32),
            'dec': dec_sources.astype(np.float32),
            'kappa': kappa_values.astype(np.float32),
            'gamma1': gamma1_values.astype(np.float32),
            'gamma2': gamma2_values.astype(np.float32),
            'eobs1': e1_obs.astype(np.float32),
            'eobs2': e2_obs.astype(np.float32),
            'weights': weights.astype(np.float32)
        })

        return gamma_table

    def compute_T17_Nz(self, z_persbin, Nz_persbin):
        """
        Computes the number of galaxies per redshift bin for the T17 (Takashi et al.) source distribution.

        Parameters:
        - z_persbin: Redshift bin edges per source bin
        - Nz_persbin: Number of galaxies per source bin

        Sets:
        - self.zbins: Array of redshift bins
        - self.N_T17_persbin: Number of galaxies per T17 redshift bin per source bin
        - self.nzbins: Number of redshift bins
        - self.Nz_persbin_weights: Normalized weights per redshift bin per source bin
        """
        # Load redshift bins from file
        zbins = np.loadtxt('nofz/nofz_takashi_zbins.dat')
          
        N_T17_persbin = []      
        for sbin in range(self.nsbins):
            zfine = []
            nzfine = []

            fine_grid = 100000
            zfine = np.linspace(z_persbin[sbin][0], z_persbin[sbin][1], fine_grid) 
        
            nzfine = np.ones(fine_grid) * Nz_persbin[sbin][0] / fine_grid 
                
            for i in range(len(z_persbin[sbin])-2):
                zfine = np.concatenate((zfine, np.linspace(z_persbin[sbin][i+1], z_persbin[sbin][i+2], fine_grid)[1:]))
                nzfine = np.concatenate((nzfine, np.ones(fine_grid-1) * Nz_persbin[sbin][i+1] / (fine_grid-1)))
                
            N_T17 = []
            for j in range(len(zbins)):
                if(zbins[j,1] > zfine[-1]):
                    N_T17.append(0)
                elif(zbins[j,2] > zfine[-1]):
                    lower_edge = np.where(zbins[j,1] > zfine)[0][-1]
                    N_T17.append(np.sum(nzfine[lower_edge:]))
                else: 
                    lower_edge = np.where(zbins[j,1] > zfine)[0][-1]
                    upper_edge = np.where(zbins[j,2] > zfine)[0][-1]
                    N_T17.append(np.sum(nzfine[lower_edge:upper_edge]))

            N_T17_persbin.append(N_T17)
            
        self.zbins = zbins
        self.N_T17_persbin = np.array(N_T17_persbin)
        self.nzbins = len(zbins)
        
        # Normalize weights
        weights = []
        for i in range(self.nsbins):
            weights.append(self.N_T17_persbin[i] / np.sum(self.N_T17_persbin[i]))
        self.Nz_persbin_weights = np.array(weights)       
    
    

    def create_gal_positions(self, los, random_seed=42):
        """
        Creates galaxy positions by sampling from the density field.

        Parameters:
        - los (int): Line of sight index
        - random_seed (int): Seed for random number generator
        """
        np.random.seed(random_seed)

        fname = self.download_dir / f'allskymap_nres{self.nres}r{los:03}.delta_shell.dat'

        mu_mean = self.N_T17_persbin / self.npix

        self.pixel_perzbin_persbin = [[] for _ in range(self.nzbins)]
        self.ra_sources_perzbin_persbin = [[] for _ in range(self.nzbins)]
        self.dec_sources_perzbin_persbin = [[] for _ in range(self.nzbins)]
        self.redshift_sources_perzbin_persbin = [[] for _ in range(self.nzbins)]

        with open(fname, 'rb') as f:
            for i in tqdm(range(self.nzbins), desc="Reading density shells"):
                kplane = int.from_bytes(f.read(4), 'little')
                delta_shell = np.frombuffer(f.read(self.npix * 4), dtype=np.float32)

                for sbin in range(self.nsbins):
                    mu = mu_mean[sbin, i] * 2 * (1 + delta_shell)
                    ngal = np.random.poisson(mu)
                    ipix = np.repeat(np.arange(self.npix), ngal)
                    if ipix.size > 0:
                        selected = np.random.choice(ipix, int(round(self.N_T17_persbin[sbin, i])), replace=False)
                        ra, dec = hp.pix2ang(nside=self.nside, ipix=selected, lonlat=True)
                        redshift = np.ones_like(ra)*self.zbins[i,0]
                    else:
                        selected, ra, dec = [], [], []
                        redshift = []

                    self.pixel_perzbin_persbin[i].append(selected)
                    self.ra_sources_perzbin_persbin[i].append(ra)
                    self.dec_sources_perzbin_persbin[i].append(dec)
                    self.redshift_sources_perzbin_persbin[i].append(redshift)

    def add_noise_sigma(self, g1, g2, epsilon_cov=None, epsilon_mean=None, epsilon_data=None, random_seed=42):
        """
        Adds noise to the shear measurements using a given covariance matrix or a KDE generator.

        Parameters:
        - g1, g2 (np.ndarray): Reduced shear components
        - epsilon_cov (np.ndarray): Covariance matrix for intrinsic ellipticity
        - epsilon_mean (np.ndarray): Mean of intrinsic ellipticity
        - epsilon_data (np.ndarray): Data for KDE estimation of intrinsic ellipticity
        - random_seed (int): Seed for random number generator

        Returns:
        - e1_obs (np.ndarray): Observed ellipticity component e1
        - e2_obs (np.ndarray): Observed ellipticity component e2
        - weights (np.ndarray): Weights associated with the samples
        """
        np.random.seed(random_seed)

        g = g1 + 1j * g2
        mask = np.abs(g) < 1

        if epsilon_data is not None:
            epsilon_generator = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(epsilon_data.T)
            eps1_eps2 = epsilon_generator.sample(len(g), random_state=random_seed)
            weights = eps1_eps2[:, 2]
            eps_int = eps1_eps2[:, 0] + 1j * eps1_eps2[:, 1]
        else:
            eps_int = np.random.multivariate_normal(epsilon_mean, epsilon_cov, len(g)).view(np.complex128).flatten()
            weights = np.ones(len(g))

        epsilon_obs = np.empty_like(eps_int)

        # Apply transformations
        epsilon_obs[mask] = (g[mask] + eps_int[mask]) / (1 + np.conj(g[mask]) * eps_int[mask])
        epsilon_obs[~mask] = (1 + g[~mask] * np.conj(eps_int[~mask])) / (np.conj(g[~mask]) + np.conj(eps_int[~mask]))

        # Ensure that observed ellipticity magnitude is less than 1
        invalid = np.abs(epsilon_obs) >= 1
        max_iterations = 10
        iterations = 0
        while np.any(invalid) and iterations < max_iterations:
            if epsilon_data is not None:
                eps1_eps2 = epsilon_generator.sample(np.sum(invalid), random_state=random_seed)
                eps_int_new = eps1_eps2[:, 0] + 1j * eps1_eps2[:, 1]
                weights[invalid] = eps1_eps2[:, 2]
            else:
                eps_int_new = np.random.multivariate_normal(epsilon_mean, epsilon_cov, np.sum(invalid)).view(
                    np.complex128).flatten()

            idx_invalid = np.where(invalid)[0]
            mask_invalid = mask[invalid]
            g_invalid = g[invalid]
            eps_obs_new = np.empty_like(eps_int_new)

            eps_obs_new[mask_invalid] = (g_invalid[mask_invalid] + eps_int_new[mask_invalid]) / (
                    1 + np.conj(g_invalid[mask_invalid]) * eps_int_new[mask_invalid])
            eps_obs_new[~mask_invalid] = (1 + g_invalid[~mask_invalid] * np.conj(eps_int_new[~mask_invalid])) / (
                    np.conj(g_invalid[~mask_invalid]) + np.conj(eps_int_new[~mask_invalid]))

            epsilon_obs[invalid] = eps_obs_new
            invalid = np.abs(epsilon_obs) >= 1
            iterations += 1

            if iterations == max_iterations:
                raise RuntimeError("Maximum iterations reached while ensuring |epsilon_obs| < 1")

        return epsilon_obs.real, epsilon_obs.imag, weights

    def combine_source_planes(self, los):
        """
        Combines the source planes by stacking the shear and convergence maps across redshift bins.

        Parameters:
        - los (int): Line of sight index
        """
        self.kappa_allbins = [[] for _ in range(self.nsbins)]
        self.gamma1_allbins = [[] for _ in range(self.nsbins)]
        self.gamma2_allbins = [[] for _ in range(self.nsbins)]
        self.ra_allbins = [[] for _ in range(self.nsbins)]
        self.dec_allbins = [[] for _ in range(self.nsbins)]
        self.redshift_allbins = [[] for _ in range(self.nsbins)]

        for zbin in tqdm(range(self.nzbins), desc="Combining source planes"):
            kappa_zbin, gamma1_zbin, gamma2_zbin = self.load_maps(z=zbin + 1, los=los)
            if np.any(np.isnan([kappa_zbin, gamma1_zbin, gamma2_zbin])):
                print(f'Lens maps have NaNs in redshift bin {zbin + 1}')

            for sbin in range(self.nsbins):
                pix_indices = self.pixel_perzbin_persbin[zbin][sbin]
                if len(pix_indices) == 0:
                    continue
                self.kappa_allbins[sbin].append(kappa_zbin[pix_indices])
                self.gamma1_allbins[sbin].append(gamma1_zbin[pix_indices])
                self.gamma2_allbins[sbin].append(gamma2_zbin[pix_indices])
                self.ra_allbins[sbin].append(self.ra_sources_perzbin_persbin[zbin][sbin])
                self.dec_allbins[sbin].append(self.dec_sources_perzbin_persbin[zbin][sbin])
                self.redshift_allbins[sbin].append(self.redshift_sources_perzbin_persbin[zbin][sbin])

        # Concatenate lists into arrays
        for sbin in range(self.nsbins):
            self.kappa_allbins[sbin] = np.concatenate(self.kappa_allbins[sbin])
            self.gamma1_allbins[sbin] = np.concatenate(self.gamma1_allbins[sbin])
            self.gamma2_allbins[sbin] = np.concatenate(self.gamma2_allbins[sbin])
            self.ra_allbins[sbin] = np.concatenate(self.ra_allbins[sbin])
            self.dec_allbins[sbin] = np.concatenate(self.dec_allbins[sbin])
            self.redshift_allbins[sbin] = np.concatenate(self.redshift_allbins[sbin])

    def create_sigma_shear_catalogue(self, los, sbin, epsilon_cov=None, epsilon_mean=None, epsilon_data=None,
                                     random_seed=42):
        """
        Creates a shear catalogue for the given source bin by adding noise to the shear maps.

        Parameters:
        - los (int): Line of sight index
        - sbin (int): Source bin index
        - epsilon_cov (np.ndarray): Covariance matrix for intrinsic ellipticity
        - epsilon_mean (np.ndarray): Mean of intrinsic ellipticity
        - epsilon_data (np.ndarray): Data for KDE estimation of intrinsic ellipticity
        - random_seed (int): Seed for random number generator

        Returns:
        - gamma_table (Table): Astropy Table containing the shear catalogue
        """
        kappa = self.kappa_allbins[sbin]
        gamma1 = self.gamma1_allbins[sbin]
        gamma2 = self.gamma2_allbins[sbin]
        ra = self.ra_allbins[sbin]
        dec = self.dec_allbins[sbin]
        redshift = self.redshift_allbins[sbin]

        # Compute reduced shear
        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)

        # Add noise to shear
        e1_obs, e2_obs, weights = self.add_noise_sigma(
            g1, g2, epsilon_cov=epsilon_cov, epsilon_mean=epsilon_mean,
            epsilon_data=epsilon_data, random_seed=random_seed
        )

        # Create Astropy Table
        gamma_table = Table({
            'ra': ra.astype(np.float32),
            'dec': dec.astype(np.float32),
            'redshift': redshift.astype(np.float32),
            'kappa': kappa.astype(np.float32),
            'gamma1': gamma1.astype(np.float32),
            'gamma2': gamma2.astype(np.float32),
            'eobs1': e1_obs.astype(np.float32),
            'eobs2': e2_obs.astype(np.float32),
            'weights': weights.astype(np.float32)
        })

        return gamma_table
