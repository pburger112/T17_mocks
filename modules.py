import numpy as np
import healpy as hp
import os
from astropy.table import Table
from pathlib import Path
from sklearn.neighbors import KernelDensity
import wget
from astropy.io import fits
from tqdm import tqdm

# =================================
#               modules
# =================================
class lensing_cov():

    def __init__(self, 
                 download_dir,
                 output_dir,
                 nres,
                 nsbins,
                 ):
        """
        Constructor
        """
        # super
        super(lensing_cov, self).__init__()

        self.download_dir = download_dir
        self.output_dir = output_dir
        self.nres = nres
        self.nside = 2**self.nres
        self.npix = hp.nside2npix(nside=self.nside)
        self.nsbins = nsbins
        
    def download_lensing_files(self, zindex, los, overwrite):
        if(los<54):
            url = f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres{self.nres}/allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat'
            filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat'
        else:
            url = f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub2/nres{self.nres}/allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat'
            filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.zs{zindex}.mag.dat'
          
        if((Path(filename).exists())&(overwrite==False)):
            print(filename, 'already exits')
        else:
            wget.download(url,out=filename,bar=wget.bar_adaptive)  
            print(filename, 'downloaded')
            
            

    def download_density_files(self, los, overwrite):
        if(los<54):
            url = f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres{self.nres}/delta_shell_maps/allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
            filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
        else:
            url = f'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub2/nres{self.nres}/delta_shell_maps/allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
            filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.delta_shell.dat'
         
        if((Path(filename).exists())&(overwrite==False)):
            print(filename, 'already exits')
        else:
            wget.download(url,out=filename,bar=wget.bar_adaptive)  
            print(filename, 'downloaded')
        
    def check_files(self,los):
        
        all_files_available = True
        for zbin in range(self.nzbins):
            filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.zs{zbin+1}.mag.dat'
    
            if(Path(filename).exists()):
                # print('lensing file '+filename+' already downloaded')
                continue
            else:
                print('downloading lensing file '+filename)
                self.download_lensing_files(zindex=zbin+1, los=los, overwrite = True)
                all_files_available = False
        if(all_files_available):
            print('all source files available')
            
        all_files_available = False 
        filename = self.download_dir+f'/allskymap_nres{self.nres:02}r{los:03}.delta_shell.dat'
        if(Path(filename).exists()):
            all_files_available = True
        else:
            print('downloading density file '+filename)
            download_density_files(los=los,overwrite = True)   

        if(all_files_available):
            print('density file '+filename+' available')

    def load_maps(self, z, los):
        
        filename = self.download_dir+f'/allskymap_nres{self.nres}r{los:03}.zs{z}.mag.dat'
        
        skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
        load_blocks = [skip[i+1]-skip[i] for i in range(0, 6)]

        with open(filename, 'rb') as f:
            rec = np.fromfile(f, dtype='uint32', count=1)[0]
            nside = np.fromfile(f, dtype='int32', count=1)[0]
            self.npix = np.fromfile(f, dtype='int64', count=1)[0]
            rec = np.fromfile(f, dtype='uint32', count=1)[0]
            #print("nside:{} self.npix:{}".format(nside, self.npix))

            rec = np.fromfile(f, dtype='uint32', count=1)[0]

            print('load kappa')
            kappa = np.array([])
            r = self.npix
            for i, l in enumerate(load_blocks):
                blocks = min(l, r)
                load = np.fromfile(f, dtype='float32', count=blocks)
                np.fromfile(f, dtype='uint32', count=2)
                kappa = np.append(kappa, load)
                r = r-blocks
                
                if r == 0:
                    break
                elif r > 0 and i == len(load_blocks)-1:
                    load = np.fromfile(f, dtype='float32', count=r)
                    np.fromfile(f, dtype='uint32', count=2)
                    kappa = np.append(kappa, load)
            print('load gamma1')
            gamma1 = np.array([])
            r = self.npix
            for i, l in enumerate(load_blocks):
                blocks = min(l, r)
                load = np.fromfile(f, dtype='float32', count=blocks)
                np.fromfile(f, dtype='uint32', count=2)
                gamma1 = np.append(gamma1, load)
                r = r-blocks
                if r == 0:
                    break
                elif r > 0 and i == len(load_blocks)-1:
                    load = np.fromfile(f, dtype='float32', count=r)
                    np.fromfile(f, dtype='uint32', count=2)
                    gamma1 = np.append(gamma1, load)

            print('load gamma2')
            gamma2 = np.array([])
            r = self.npix
            for i, l in enumerate(load_blocks):
                blocks = min(l, r)
                load = np.fromfile(f, dtype='float32', count=blocks)
                np.fromfile(f, dtype='uint32', count=2)
                gamma2 = np.append(gamma2, load)
                r = r-blocks
                if r == 0:
                    break
                elif r > 0 and i == len(load_blocks)-1:
                    load = np.fromfile(f, dtype='float32', count=r)
                    np.fromfile(f, dtype='uint32', count=2)
                    gamma2 = np.append(gamma2, load)

            return kappa,gamma1,gamma2

                

    def rotate_gals(self, ras, decs, gammas1, gammas2, rotangle, inv=False, units="deg"):
        """ Rotates survey patch s.t. its center of mass lies in the origin. """
        
        # Map (ra, dec) --> (theta, phi)
        if units=="deg":
            decs_rad = decs * np.pi/180.
            ras_rad = ras * np.pi/180.
            rotangle_rad = rotangle * np.pi/180.
        thetas = np.pi/2. + decs_rad
        phis = ras_rad
        
        # Compute rotation angle
        thisrot = hp.Rotator(rot=[0,rotangle_rad], deg=False, inv=inv)
        rotatedthetas, rotatedphis = thisrot(thetas,phis, inv=False)

        gamma_rot = (gammas1+1J*gammas2) * np.exp(1J * 2 * thisrot.angle_ref(rotatedthetas, rotatedphis,inv=True))
        
        # Transform back to (ra,dec)
        ra_rot = rotatedphis
        dec_rot = rotatedthetas - np.pi/2.
        if units=="deg":
            dec_rot *= 180./np.pi
            ra_rot *= 180./np.pi
        
        return ra_rot, dec_rot, gamma_rot.real, gamma_rot.imag


    def create_lensing_maps(self,los):

        self.npix = hp.nside2self.npix(self.nside)
        kappa_allbins = np.array([np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix)])
        gamma1_allbins = np.array([np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix)])
        gamma2_allbins = np.array([np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix),np.zeros(self.npix)])
        for zbin in range(self.nzbins):
            print('loading ',zbin+1,los)
            kappa_zbin,gamma1_zbin,gamma2_zbin=self.load_maps(z=zbin+1,los=los)
            for sbin in range(self.nsbins):
                kappa_allbins[sbin-1]=kappa_allbins[sbin-1]+kappa_zbin*self.Nz_persbin_weights[sbin][zbin]
                gamma1_allbins[sbin-1]=gamma1_allbins[sbin-1]+gamma1_zbin*self.Nz_persbin_weights[sbin][zbin]
                gamma2_allbins[sbin-1]=gamma2_allbins[sbin-1]+gamma2_zbin*self.Nz_persbin_weights[sbin][zbin]
            
            del kappa_zbin
            del gamma1_zbin
            del gamma2_zbin

        for sbin in range(self.nsbins):           
            hp.fitsfunc.write_map(self.output_dir+'/gamma1_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin+1)+'_'+str(los)+'.fits', gamma1_allbins[sbin],dtype=np.float32,overwrite=True)
            hp.fitsfunc.write_map(self.output_dir+'/gamma2_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin+1)+'_'+str(los)+'.fits', gamma2_allbins[sbin],dtype=np.float32,overwrite=True)
            hp.fitsfunc.write_map(self.output_dir+'/kappa_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin+1)+'_'+str(los)+'.fits', kappa_allbins[sbin],dtype=np.float32,overwrite=True)

        print('finished writing shear maps ',los)
        
    def loading_shear_maps(self,sbin,los):

        gamma1 = hp.read_map(self.output_dir+'/gamma1_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin)+'_'+str(los)+'.fits')
        gamma2 = hp.read_map(self.output_dir+'/gamma2_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin)+'_'+str(los)+'.fits')
        kappa = hp.read_map(self.output_dir+'/kappa_nside'+str(self.nside)+'_KiDS1000_tomobin'+str(sbin)+'_'+str(los)+'.fits')
            
        print('finished loading shear maps ',los,sbin)

        return gamma1,gamma2,kappa
    
    
    def add_noise_obs(self, g1, g2, eps1, eps2):
        # Apply reduced shear and check which trafo to use
        g = g1+1J*g2
        toobig = np.abs(g)>=1

        eps_int = eps1 + 1J*eps2
        eps_int = eps_int*np.exp(-2*1j*np.random.uniform(0,2*np.pi,size=len(eps_int)))
    
        epsilon_obs = np.ones(len(eps1))+1j*np.ones(len(eps1))
        epsilon_obs[~toobig] = (g[~toobig] + eps_int[~toobig])/(1 + g[~toobig].conj()*eps_int[~toobig])
        epsilon_obs[toobig] = (1 + g[toobig]*eps_int[toobig].conj())/(g[toobig].conj() + eps_int[toobig].conj())
        
        return epsilon_obs.real, epsilon_obs.imag  
    
    
    def create_real_shear_catalogue(self, sbin, los, ra_sources, dec_sources, e_1_rot, e_2_rot, weights):

        gamma1,gamma2,kappa  = self.loading_shear_maps(sbin,los)
        
        pix_center_rot = hp.ang2pix(nside=self.nside,theta=ra_sources,phi=dec_sources,lonlat=True)

        kappa_center_values = kappa[pix_center_rot]
        gamma1_center_values = gamma1[pix_center_rot]
        gamma2_center_values = gamma2[pix_center_rot]
        
        g1,g2 = gamma1/(1-kappa),gamma2/(1-kappa)
        
        e1_obs,e2_obs=self.add_noise_obs(g1=g1, g2=g2, eps1=e_1_rot, eps2=e_2_rot)

        gamma_table = Table()
        gamma_table.add_column(ra_sources.astype(np.float32),name=r'ra')
        gamma_table.add_column(dec_sources.astype(np.float32),name=r'dec')
        gamma_table.add_column(kappa_center_values.astype(np.float32),name=r'kappa center')
        gamma_table.add_column(gamma1_center_values.astype(np.float32),name=r'gamma1 center')
        gamma_table.add_column(gamma2_center_values.astype(np.float32),name=r'gamma2 center')
        gamma_table.add_column(e1_obs.real.astype(np.float32),name=r'eobs1 center')
        gamma_table.add_column(e2_obs.imag.astype(np.float32),name=r'eobs2 center')
        gamma_table.add_column(weights.astype(np.float32),name=r'weights')
        
        return gamma_table
    
    
    def compute_T17_Nz(self,z_persbin,Nz_persbin):
    
        zbins = np.loadtxt('nofz/nofz_takashi_zbins.dat')
          
        N_T17_persbin = []      
        for sbin in range(self.nsbins):
            zfine = []
            nzfine = []

            fine_grid = 100000
            zfine = np.linspace(z_persbin[sbin][0],z_persbin[sbin][1],fine_grid) 
            print(sbin)
        
            nzfine = np.ones(fine_grid)*Nz_persbin[sbin][0]/fine_grid 
                
            for i in range(len(z_persbin[sbin])-2):
                zfine = np.concatenate((zfine,np.linspace(z_persbin[sbin][i+1],z_persbin[sbin][i+2],fine_grid)[1:]))
                nzfine = np.concatenate((nzfine,np.ones(fine_grid-1)*Nz_persbin[sbin][i+1]/(fine_grid-1)))
                
            N_T17 = []
            for j in range(len(zbins)):
                if(zbins[j,1]>zfine[-1]):
                    N_T17.append(0)
                elif(zbins[j,2]>zfine[-1]):
                    lower_edge = np.where(zbins[j,1]>zfine)[0][-1]
                    N_T17.append(np.sum(nzfine[lower_edge:]))
                else: 
                    lower_edge = np.where(zbins[j,1]>zfine)[0][-1]
                    upper_edge = np.where(zbins[j,2]>zfine)[0][-1]
                    N_T17.append(np.sum(nzfine[lower_edge:upper_edge]))

            N_T17_persbin.append(N_T17)
            
        self.zbins = zbins
        self.N_T17_persbin = np.array(N_T17_persbin)
        self.nzbins = len(zbins)
        
        weights = []
        for i in range(self.nsbins):
            weights.append(self.N_T17_persbin[i]/np.sum(self.N_T17_persbin[i]))
        self.Nz_persbin_weights = np.array(weights)       


    def create_gal_positions(self,los, random_seed = 42):
        
        np.random.seed(random_seed)
        
        fname = self.download_dir+f'/allskymap_nres{self.nres:02}r{los:03}.delta_shell.dat'
        
        mu_mean = self.N_T17_persbin/self.npix

        self.pixel_perzbin_persbin = []
        self.ra_sources_perzbin_persbin = []
        self.dec_sources_perzbin_persbin = []

        with open(fname, 'rb') as f:
            for i in tqdm(range(self.N_T17_persbin.shape[1])):
                kplane = int.from_bytes(f.read(4), 'little')  # Read kplane
                delta_shell = np.frombuffer(f.read(self.npix * 4), dtype=np.float32)  # Read delta_shell
                
                pixel_perzbin = []
                ra_sources_perzbin = []
                dec_sources_perzbin = []

                for sbin in range(self.nsbins):
                    mu = mu_mean[sbin][i] * 2 * (1.0 + 1.0 * delta_shell)
                    ngal = np.random.poisson(mu)
                    ipix = np.where(ngal>0)[0]
                    if(len(ipix)>0):
                        ipix = np.repeat(ipix, ngal[ipix])
                        # print(len(ipix),self.N_T17_persbin[sbin][i])
                        ipix = np.random.choice(range(len(ipix)),int(round(self.N_T17_persbin[sbin][i],0)),replace=False)
                        
                        ra,dec = hp.pix2ang(nside=2**12,ipix=ipix,lonlat=True)
                        
                        pixel_perzbin.append(ipix)
                        ra_sources_perzbin.append(ra)
                        dec_sources_perzbin.append(dec)
                    else:
                        pixel_perzbin.append([])
                        ra_sources_perzbin.append([])
                        dec_sources_perzbin.append([])
                        
                self.pixel_perzbin_persbin.append(pixel_perzbin) 
                self.ra_sources_perzbin_persbin.append(ra_sources_perzbin) 
                self.dec_sources_perzbin_persbin.append(dec_sources_perzbin) 
       
    
                     
    
    def add_noise_sigma(self, g1, g2, epsilon_cov, epsilon_mean, epsilon_generator, random_seed = 42):
        
        np.random.seed(random_seed)
        
        g = g1+1j*g2
        g_bigger1 = np.abs(g)>=1
        g_smaller1 = np.abs(g)<1
        
        if(epsilon_generator):
            epsilon1,epsilon2,weight_bigger1 = epsilon_generator.sample(np.sum(g_bigger1),random_state=random_seed).T
            eps_int_bigger1 = epsilon1+1j*epsilon2

            epsilon1,epsilon2,weight_smaller1 = epsilon_generator.sample(np.sum(g_smaller1),random_state=random_seed).T
            eps_int_smaller1 = epsilon1+1j*epsilon2
        else:
            epsilon1,epsilon2,weight_bigger1 = np.random.multivariate_normal(epsilon_mean,epsilon_cov,np.sum(g_bigger1))
            eps_int_bigger1 = epsilon1+1j*epsilon2

            epsilon1,epsilon2,weight_smaller1 = np.random.multivariate_normal(epsilon_mean,epsilon_cov,np.sum(g_smaller1))
            eps_int_smaller1 = epsilon1+1j*epsilon2

        epsilon_obs = np.ones(len(g1))+1j*np.ones(len(g2))

        epsilon_obs[g_smaller1] = (g[g_smaller1] + eps_int_smaller1)/(1 + g[g_smaller1].conj()*eps_int_smaller1)
        epsilon_obs[g_bigger1] = (1 + g[g_bigger1]*eps_int_bigger1.conj())/(g[g_bigger1].conj() + eps_int_bigger1.conj())
        weights = np.ones(len(g1))
        weights[g_smaller1] = weight_smaller1
        weights[g_bigger1] = weight_bigger1
        
        while(np.sum(np.abs(epsilon_obs)>=1)>0):
            print(np.sum(np.abs(epsilon_obs)>=1))
            eobs_bigger1 = np.abs(epsilon_obs[g_bigger1])>=1
            eobs_smaller1 = np.abs(epsilon_obs[g_smaller1])>=1
            
            if(epsilon_generator):
                epsilon1,epsilon2,weight_bigger1 = epsilon_generator.sample(np.sum(g_bigger1),random_state=random_seed).T
                eps_int_bigger1 = epsilon1+1j*epsilon2

                epsilon1,epsilon2,weight_smaller1 = epsilon_generator.sample(np.sum(g_smaller1),random_state=random_seed).T
                eps_int_smaller1 = epsilon1+1j*epsilon2
            else:
                epsilon1,epsilon2,weight_bigger1 = np.random.multivariate_normal(epsilon_mean,epsilon_cov,np.sum(g_bigger1))
                eps_int_bigger1 = epsilon1+1j*epsilon2

                epsilon1,epsilon2,weight_smaller1 = np.random.multivariate_normal(epsilon_mean,epsilon_cov,np.sum(g_smaller1))
                eps_int_smaller1 = epsilon1+1j*epsilon2
                
            epsilon_obs[np.where(g_smaller1)[0][eobs_smaller1]] = (g[g_smaller1][eobs_smaller1] + eps_int_smaller1)/(1 + g[g_smaller1][eobs_smaller1].conj()*eps_int_smaller1)
            epsilon_obs[np.where(g_bigger1)[0][eobs_bigger1]] = (1 + g[g_bigger1][eobs_bigger1]*eps_int_bigger1.conj())/(g[g_bigger1][eobs_bigger1].conj() + eps_int_bigger1.conj())

            weights[g_smaller1] = weight_smaller1
            weights[g_bigger1] = weight_bigger1
        
        return epsilon_obs.real, epsilon_obs.imag, weights 
    
    
    
    
    def combine_sourceplanes(self,los):
        
        self.kappa_allbins  = [[] for i in range(self.nsbins)]
        self.gamma1_allbins  = [[] for i in range(self.nsbins)]
        self.gamma2_allbins  = [[] for i in range(self.nsbins)]
        self.ra_allbins  = [[] for i in range(self.nsbins)]
        self.dec_allbins  = [[] for i in range(self.nsbins)]
        
        for zbin in range(self.nzbins):
            print('loading ',zbin+1,los)
            kappa_zbin,gamma1_zbin,gamma2_zbin=self.load_maps(z=zbin+1,los=los)
            for sbin in range(self.nsbins):
                    pix_center = self.pixel_perzbin_persbin[zbin][sbin]
                    self.kappa_allbins[sbin] = np.concatenate((self.kappa_allbins[sbin],kappa_zbin[pix_center]))
                    self.gamma1_allbins[sbin] = np.concatenate((self.gamma1_allbins[sbin],gamma1_zbin[pix_center]))
                    self.gamma2_allbins[sbin] = np.concatenate((self.gamma2_allbins[sbin],gamma2_zbin[pix_center]))
                    self.ra_allbins[sbin] = np.concatenate((self.ra_allbins[sbin],self.ra_sources_perzbin_persbin[zbin][sbin]))
                    self.dec_allbins[sbin] = np.concatenate((self.dec_allbins[sbin],self.dec_sources_perzbin_persbin[zbin][sbin]))

    
    def create_sigma_shear_catalogue(self, los, sbin, epsilon_cov=None, epsilon_mean=None, epsilon_data=[], random_seed = 42):
        
        gamma_table = Table()     
        
        kappa = self.kappa_allbins[sbin]
        gamma1 = self.gamma1_allbins[sbin]
        gamma2 = self.gamma2_allbins[sbin]
        ra = self.ra_allbins[sbin]
        dec = self.dec_allbins[sbin]
        
        g1,g2 = gamma1/(1-kappa),gamma2/(1-kappa)
        
        if(len(epsilon_data)>0):
            epsilon_generator = KernelDensity(kernel='gaussian', bandwidth=0.001)
            epsilon_generator.fit(epsilon_data.T)
        else:
            epsilon_generator = None

        e1_obs,e2_obs=self.add_noise_sigma(g1=g1, g2=g2, epsilon_cov=epsilon_cov, epsilon_mean=epsilon_mean, epsilon_generator=epsilon_generator, random_seed=random_seed)
    
        gamma_table.add_column(ra.astype(np.float32),name=r'ra sbin'+str(sbin))
        gamma_table.add_column(dec.astype(np.float32),name=r'dec sbin'+str(sbin))
        gamma_table.add_column(kappa.astype(np.float32),name=r'kappa sbin'+str(sbin))
        gamma_table.add_column(gamma1.astype(np.float32),name=r'gamma1 sbin'+str(sbin))
        gamma_table.add_column(gamma2.astype(np.float32),name=r'gamma2 sbin'+str(sbin))
        gamma_table.add_column(e1_obs.real.astype(np.float32),name=r'eobs1 sbin'+str(sbin))
        gamma_table.add_column(e2_obs.imag.astype(np.float32),name=r'eobs2 sbin'+str(sbin))

        
        return gamma_table



