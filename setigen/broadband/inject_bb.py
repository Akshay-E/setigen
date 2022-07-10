import os
import shutil 

GPU_FLAG = os.getenv('SETIGEN_ENABLE_GPU', '0')

if GPU_FLAG == '1':
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp
    
import numpy as np
import astropy.units as u
import warnings
import scipy
#from setigen.voltage import raw_utils
from ..voltage import raw_utils
#import setigen as stg 
import logging
import matplotlib.pyplot as plt
'''
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='broadband_log.log', mode='w')
logger.addHandler(fhandler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
fhandler.setFormatter(formatter)
level_log = logging.INFO
logging.basicConfig( level=level_log)
'''
logging.basicConfig(filename="setigen_bb.log",
                    format='%(asctime)s - %(levelname)s - %(lineno)s - %(message)s',
                    filemode='w')

logger = logging.getLogger() 
logger.setLevel(logging.INFO)

class inj_broadband(object):
    
    def __init__(self, 
                 input_file_stem,
                 pulse_time,
                 dm=100, 
                 width=1000, 
                 snr=10      
                ):
        """
        Parameters
        ----------
        input_file_stem : str
            Filename or path stem to input RAW data 
        pulse_time : float
            Start time for injected signal
        dm : float, optional
            Dispersion measure. The default is 100.
        width : int, optional
            Total span of the pulse profile to be generated in samples. The default is 1000.
            If custom pulse profile is used, width is calculated accordingly.
        snr : float, optional
            Desired signal to noise ratio of injected signal at raw voltage level. The default is 10.

        Returns
        -------
        None.

        """
        self.raw_params = raw_utils.get_raw_params(input_file_stem=input_file_stem)
        self.num_pols=int(self.raw_params['num_pols'])
        self.raw_params['fch1'], self.raw_params['chan_bw'], self.raw_params['center_freq'] = self.raw_params['fch1']/1e6, self.raw_params['chan_bw']/1e6, self.raw_params['center_freq']/1e6
        self.obs_bw= self.raw_params['num_chans'] * self.raw_params['chan_bw']
        if self.raw_params['ascending']:
            self.f_low= self.raw_params['fch1']
        else:
            self.f_low= self.raw_params['fch1'] + self.obs_bw
        
        self.input_file_stem=input_file_stem
        self.dm=dm
        self.width=int(width)
        self.snr=snr
        self.D=4.148808e3
        self.pulse_time=float(pulse_time)
        
        assert self.pulse_time > 0 , f"injection time cannot be 0"
        assert self.pulse_time < self.raw_params['obs_length'], f"injection time cannot be greater than length of file( {self.raw_params['obs_length']}) "
        
        self.calc_smear() 
        self.adjust_time()
        self.file_params()
        
        print(f" Adjusted injection time for channel {self.f_low + abs(self.obs_bw)} MHz {round(self.adjusted_pulse_time,3)}")
        # logger.info("Object initialisation passed")
        
    
    def calc_smear(self,x=2):
        """
        Calculate smearing in the band based on the desired power law.

        Parameters
        ----------
        x : float, optional
            Desired power law. The default is 2 (Natural dispersion).

        Returns
        -------
        None.
        """
        
        f_end=self.f_low + abs(self.obs_bw)
        self.smear= self.D*self.dm*(self.f_low**-x - f_end**-x)
        
    
    def adjust_time(self):
        """
        Adjust the start time for injected signal to place it at the starting of corresponding block.

        Note
        -------
        If smearning across the band from adjusted pulse time exceeds length of the file, the function will try to re-adjust the start time to accomodate the smearing. 
        Will raise an error if re-adjusting also fails.

        """
        
        self.blocks_per_file = raw_utils.get_blocks_per_file(self.input_file_stem)
        self.time_per_block = self.raw_params['block_size'] / (self.raw_params['num_antennas'] * self.raw_params['num_chans'] * (2 * self.raw_params['num_pols'] * self.raw_params['num_bits'] // 8)) * self.raw_params['tbin']

        self.blocks_to_read=int( np.ceil(self.smear/self.time_per_block))
        self.start_block= int(np.ceil(self.pulse_time/self.time_per_block))
        
        self.adjusted_pulse_time=(self.start_block - 1) * self.time_per_block 

        try :
            
            assert self.adjusted_pulse_time + self.smear < self.raw_params['obs_length'] 
        
        except AssertionError:
            
            self.pulse_time=self.raw_params['obs_length'] - self.smear
            
            assert self.pulse_time  > 0 , f"Smearing across the band ({round(self.smear, 3)}) exceeds length of file ({round(self.raw_params['obs_length'], 3)}) from adjusted pulse time ({round(self.adjusted_pulse_time, 2)}). Try changing DM. "
            
            warnings.warn(f"Smearing across the band ({round(self.smear, 3)}) exceeds length of file ({round(self.raw_params['obs_length'], 3)}) from adjusted pulse time ({round(self.adjusted_pulse_time, 2)}). Re-adjusting injection time. ")
            
            self.adjust_time()
            
            logger.warning("Initial adjusted time failed. Re-adjusted time to account the smearing. Try reducing DM or pulse time to prevent the warning")
            

    def file_params(self):
        """
        Calculate basic file parameters, header_size, block_size, channel_size, (block_size+header_size)

        """
        header=raw_utils.read_header(f'{self.input_file_stem}.0000.raw')
        
        self.header_size= int(512 * np.ceil((80 * (len(header) + 1)) / 512))
        self.data_size= int(self.raw_params['block_size'])
        self.chan_size=int(self.data_size/self.raw_params['num_chans'])
        self.block_read_size =  self.header_size + self.data_size
        
        
    def info(self):
        """
        Display useful parameters for the RAW file

        """
        print("\nRaw parameters from header\n")
        for key,value in self.raw_params.items():
            print(key, ' : ', value)
        
        print("\nOther useful parameters\n")
        tag=['Smearing across band(s)','Blocks_per_file', 'Blocks_to_read' , 'Start_block' , 'Time_per_block','File_length(s)','Adjusted_pulse_time']
        value=(self.smear,self.blocks_per_file,self.blocks_to_read , self.start_block , self.time_per_block,self.time_per_block*self.blocks_per_file,self.adjusted_pulse_time)
        
        for i in range(len(tag)):
            print(tag[i], ':', value[i])
            
#Dispersion by Convolution
    
    def imp_res(self, imp_length):
        """
        Computes impulse response of ISM for each channel.

        Parameters
        ----------
        imp_length : int
            Impulse response length in samples.

        Returns
        -------
        H : numpy.ndarray
            2D array of impulse response for the data

        """
        logger.info(f"Generating impulse response array for {self.raw_params['num_chans']} channels")
        
        f_coarse_dev=xp.linspace(0,np.abs(self.obs_bw),self.raw_params['num_chans'], endpoint=False)
        K=2 * np.pi * self.D *1e6 * self.dm / self.f_low**2
        fl0 = xp.linspace(0,np.abs(self.raw_params['chan_bw']), imp_length, endpoint=False)
        H=np.empty((self.raw_params['num_chans'], imp_length), dtype=complex)
        
        if GPU_FLAG=='1':
            try:
                for i in range(self.raw_params['num_chans']):
                    fl=fl0 + f_coarse_dev[i]
                    V=(fl**2/(fl + self.f_low))
                    H[i]= xp.asnumpy( xp.fft.ifft ( xp.exp(1j * K * (V ) )))

            except AttributeError:
                logger.debug(f"Recommended to install CuPy. This is not necessary for injection, but will highly accelerate the pipeline")
                fl=fl0 + f_coarse_dev[:, None]
                V=(fl**2/(fl + self.f_low))
                H=scipy.fft.ifft( xp.exp(1j* K * V ) )
        else:
            logger.debug("GPU not enabled. Set (os.environ['SETIGEN_ENABLE_GPU'] = '1') in Python to enable GPU")
            fl=fl0 + f_coarse_dev[:, None]
            V=(fl**2/(fl + self.f_low))
            H=scipy.fft.ifft( xp.exp(1j* K * V ) )
        if not self.raw_params['ascending']:
            H=np.flip(H, axis=0)
                    
        logger.info(f"Generated impulse response.")
        
        return(H)
        
    
    def pad(self, L, il):
        """
        Function to pad data with il-1 samples. Padding prevents the boundary effect of convolution.

        Parameters
        ----------
        L : numpy.ndarray
            Complex time series.
        il : int
            Length of impulse response.

        Returns
        -------
        padded_block : numpy.ndarray
            Padded complex time series.
            
        Note
        ----
        If (il-1) samples are not available for padding, padding is aborted and convolution mode is returned as 'same'.
        Upon successful padding convolution mode is returned as 'valid'.
        """
        pad_start_block= self.start_block - self.blocks_to_read
        
        if pad_start_block<1:
            
            logger.warning(f"Incomplete data points for padding( impulse_len-1 samples from previous blocks is required). Boundary effects of convolution will be visible in the injected signal. Try increasing the start time of the pulse to prevent the warning.   ")
            warnings.warn('Incomplete data points for padding. Boundary effects of convolution will be visible.')
            mode='same'
            return(L, mode)
       
        else:
            self.file_handler.seek((pad_start_block-1) * self.block_read_size, 0)
            cmplx_data=self.collect_data(pad_start_block)
            
            M_1= cmplx_data[:, self.num_pols * (1-il):]
            
            padded_block=np.hstack((M_1, L) )
            mode='valid'
            logger.info("Padding complete")
            return(padded_block, mode)
    

    def disperse(self, op_dir=None, profile=None, plot=False, plot_alt=None):
        """
        Function to collect raw voltage data, inject a natural broadband signal by convolving the complex time series with impulse response of ISM
        and output to a GUPPI RAW file.
        The function collects the complex data and pads it with (impulse_length-1) samples to prevent boundary effects of convolution.


        Parameters
        ----------
        op_dir : str, optional
            Full path to the output directory. Filename will be automatically appended. Default is current working directory.
        profile : array, optional
            User generated pulse prolfile. The default pulse profile generated is a single gaussian spanning the input width and peak at input snr.
        plot : bool, optional
            Display the plot of simulated pulse, impulse response and convolved output of desired channels. The default is False.
        plot_alt : int, optional
            If plot is True then display plot of every 'plot_alt' alternate channels. Default is 1, which shows output plots of every channel.

        Returns
        -------
        None.
        
        Note
        ----
        If (impulse_length-1) samples are not available for padding, boundary effect of convolution will be present in the output data(Visible as 
        a wedge like shape in the intensity domain ). 
        To prevent this try injections away from initial blocks of the file. 
        Injections with high DMs and injected pulse time near beginning of the file are susceptible to this effect. 
        
        """
        
        path, pulse_profile= self.dispatcher(op_dir, profile)
        impulse_len=int(np.ceil(self.smear/self.raw_params['tbin']))
        
        logger.info(f"impulse length in samples: {impulse_len}")

        with open(path, 'r+b') as self.file_handler:
            
            self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)
            
            logger.info(f"File handler starting from {self.file_handler.tell()}")
    
            block_cmplx=self.collect_data()
            shape = block_cmplx.shape
            
            logger.info(f"Adding pulse profile in series  ")
            
            if self.start_block - self.blocks_to_read<1:
                s= (block_cmplx.shape[1])//(2 * self.num_pols)
                e= s+ self.num_pols * self.width
                block_cmplx[ :,s:e:self.num_pols ]*=pulse_profile
            else:
                block_cmplx[ :, :self.num_pols*self.width:self.num_pols ]*=pulse_profile
                
            logger.info(f"Padding complex time series with (impulse_length-1) samples.")
            block_cmplx, conv_mode=self.pad(block_cmplx, impulse_len)
            
            h=self.imp_res( impulse_len)
            
            logger.info(f"Shape before padding {shape}. Shape after padding {block_cmplx.shape}. Shape will be same if incomplete data points for padding.")
            logger.info(f"Performing convolution of complex time series with impulse response of ISM")
    
            if self.num_pols==2:
                
                dispersed_ts=np.empty((shape), dtype=complex)
                p1=self.convolve(block_cmplx[:,::2 ], h, conv_mode)
                dispersed_ts[:,::2 ]=p1
                dispersed_ts[:,1::2 ]=p1
                
            else:
                
                dispersed_ts=self.convolve(block_cmplx, h, conv_mode)
                
            if plot:
                if plot_alt is None:
                    plot_alt=1
        
                for k in range(0,self.raw_params['num_chans'],plot_alt):
                
                    plt.figure(figsize = (10,6))
                    a1=plt.subplot(311)
                    a1.title.set_text(f'Generated pulse+padding, channel {k}')
                    a1.plot(block_cmplx[k].real)

                    a2=plt.subplot(312)
                    a2.title.set_text('Impulse response of ISM')
                    a2.plot( h[k].real)

                    a3=plt.subplot(313)
                    a3.title.set_text('Dispersed pulse')
                    a3.plot(dispersed_ts[k].real)
                    
                    plt.setp((a1,a2,a3), xticks=[])
                    plt.show()
            
            self.write_blocks(dispersed_ts)
            
            logger.info(f"Dispersion by convolution complete!")
            
            
    def convolve(self, data_cmplx, response, mode):
        """
        Function to convolve complex time series with impulse response of ISM

        Parameters
        ----------
        data_cmplx : numpy.ndarray
            Complex time series
        response : numpy.ndarray
            Impulse response
        mode : str
            Convolution mode
            mode ‘same’ returns output of length max(data_cmplx, response). Boundary effects are visible.
            mode ‘valid’ returns output of length max(data_cmplx, response) - min(data_cmplx, response) + 1. The convolution product is only given for points where the signals overlap completely.

        Returns
        -------
        dispersed_ts : nump.ndarray
            Convolved complex output 

        """
        logger.info("Performing convolution")
        if GPU_FLAG==1:
            try:
                from cupyx.scipy import signal

                if mode=='same':
                    dispersed_ts=np.empty((data_cmplx.shape), dtype=complex)
                else:
                    dispersed_ts=np.empty((data_cmplx.shape[0], data_cmplx.shape[1] - response.shape[1]+1), dtype=complex) 
                for i in range(self.raw_params['num_chans']):
                    dispersed_ts[i]=xp.asnumpy(signal.fftconvolve(xp.array(data_cmplx[i]), xp.array(response[i]), mode=mode))    

            except ImportError:
                logger.debug(f"Recommended to install CuPy. This is not necessary for injection, but will highly accelerate the pipeline")
                dispersed_ts=scipy.signal.fftconvolve(data_cmplx, response, mode=mode, axes=1)
        else:
            logger.debug(f"GPU not enabled. Set (os.environ['SETIGEN_ENABLE_GPU'] = '1') in Python to enable GPU")
            dispersed_ts=scipy.signal.fftconvolve(data_cmplx, response, mode=mode, axes=1)

        logger.info(f"Convolution complete")
        return(dispersed_ts)    


# Dispersion by sample shifting 

    def chan_time_delay(self,x):
        """
        Compute time delays relative to the high frequency channel in the data.

        Parameters
        ----------
        x : float
            Desire power law.

        Returns
        -------
        sample_to_shift : numpy.ndarray
                          time delay in samples of each channel relative to high frequency channel.   

        """
        logger.info("Computing time delays")
        
        f_chan_arr= self.f_low+np.linspace(0, abs(self.obs_bw), self.raw_params['num_chans'], endpoint=False )
        chan_centr=f_chan_arr+abs(self.raw_params['chan_bw']/2.0)
        
        time_delay= self.D*self.dm*(chan_centr**-x - chan_centr[-1]**-x)
        samples_to_shift=np.ceil(time_delay/self.raw_params['tbin'])
        
        return(samples_to_shift)

    
    def sample_shift(self, x=2 , b_type='N', op_dir=None, profile=None):
        """
        Function to collect raw voltage data, inject natural or artificial broadband signal and output to a GUPPI RAW file.

        Parameters
        ----------
        x : float, optional
            Desired power law. The default is 2(Corresponding to natural dispersion).
        b_type : str, optional
            inject any one of the 4 broadband signals;1 Natural signal and 3 Artificial signals (generated by flipping either time axis, frequency axis or both).
            The default is 'N'.
            'N': Broadband signal with natural dispersion
            'A1', 'A2', 'A3': Broadband signal with artificial dispersion (flipped time axis, frequency axis or both)
        op_dir : str, optional
            Full path to the output directory. Filename will be automatically appended. Default is current working directory.
        profile : array, optional
            User generated pulse prolfile. The default pulse profile generated is a single gaussian spanning the input width and peak at input snr.

        Raises
        ------
        Exception
            When b_type is invalid

        Returns
        -------
        None.
        
        Notes
        -----
        x can be changed to generate signals with artificial power law. b_type can be changed to generate 4 type of broadband signal. 
        Different combination of x and b_type can be tried.

        """
        
        path, pulse_profile= self.dispatcher(op_dir, profile)
        chan_flip=False
        
        if b_type=='N':
            flip = not self.raw_params['ascending']
        elif b_type=='A1':
            flip = self.raw_params['ascending']
        elif b_type=='A2':
            chan_flip, flip = not chan_flip, self.raw_params['ascending']
        elif b_type=='A3':
            chan_flip, flip = not chan_flip, not self.raw_params['ascending']
        else:
            raise Exception("Invalid plot type ")
            
        if x!=2:
            self.calc_smear(x)
            self.adjust_time()
        
        td_in_samps=self.chan_time_delay(x)
        
        samples_per_chan=int(self.chan_size/(2*self.num_pols))

        if flip:  
            td_in_samps=np.flip(td_in_samps)
        if chan_flip:
            td_in_samps=samples_per_chan*self.blocks_to_read - td_in_samps -self.width
        
        with open(path, 'r+b') as self.file_handler:

            self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)
            
            logger.info(f"File handler starting from {self.file_handler.tell()}")

            block_cmplx=self.collect_data()

            if self.num_pols==2:
                logger.info(f"Generating delays in POL1")
                block_cmplx[:,::2]= self.multiply_profile(block_cmplx[:,::2], pulse_profile, td_in_samps)
                
                logger.info(f"Generating delays in POL2")
                block_cmplx[:,1::2]= self.multiply_profile(block_cmplx[:,1::2], pulse_profile, td_in_samps)
                
            else:
                logger.info(f"Generating delays ")
                block_cmplx= self.multiply_profile(block_cmplx, pulse_profile, td_in_samps)
                                    
            self.write_blocks(block_cmplx)
            logger.info(f"Dispersion by sample shifting complete!")
    
    
    def multiply_profile(self, data, profile, delay):
        """
        Function to multiply complex time series with pulse profile according to time delay in samples between channels.

        Parameters
        ----------
        data : numpy.ndarray
            Array of raw complex voltages
        
        profile : array
            The pulse profile

        delay : array
            Time delay in sample between channels

        Returns
        -------
        data : numpy.ndarray
            Array of raw complex voltages with the injected broadband signal.  
        """
        
        for i in range(self.raw_params['num_chans']):
            s= int(delay[i])
            e= int(delay[i]+self.width)
            data[i][s:e]*=profile

        return(data)
            
#Common Functions
            
    def dispatcher(self, op_dir, profile):
        """
        A common function to generate default pulse profile if not provided by the user, append the output path and duplicate the file. 

        Parameters
        ----------
        op_dir : str, optional
            Full path to the output directory. Filename will be automatically appended. Default is current working directory.

        profile : array, optional
            User generated pulse prolfile. The default pulse profile generated is a single gaussian spanning the input width and peak at input snr.

        Returns
        -------
        path : str
            Output file stem (output directory + file stem)
        profile : array
            Pulse profile

        """
        
        if profile is None:
            profile=inj_broadband.gauss(a=self.snr, width=self.width)
            logger.info("Created default pulse profile")
        else:
            self.width=len(profile)
            logger.info("User defined pulse profile")
        
        if op_dir is None:
            path=os.path.join(os.getcwd(), f"{self.input_file_stem.split('/')[-1]}_dispersed.0000.raw")
        else:
            path=os.path.join(op_dir, f"{self.input_file_stem.split('/')[-1]}_dispersed.0000.raw")
        
        shutil.copyfile(f'{self.input_file_stem}.0000.raw', path )
        logger.info("Created duplicate file")
        
        return(path, profile)
    
    
    def collect_data(self, _from=None, to=None):
        """
        Function to collect raw voltage data from input RAW file

        Parameters
        ----------
        _from : int, optional
            Index of block to start reading from. Default is start_block calculated during object initialisation.
        to : int, optional
            Total blocks to read. Default is blocks_to_read calculated during object initialisation.

        Returns
        -------
        block_cmplx : numpy.ndarray
            2D complex array reshaped in (channel x data) format 

        """
        
        if _from is None:
            _from=self.start_block
        if to is None:
            to=self.blocks_to_read
            
        logger.info(f"Collecting data from block {_from} to {_from+to-1}, ({to} block/s)")
        
        block= (np.frombuffer(self.file_handler.read(self.block_read_size),offset=self.header_size
                              , dtype=np.int8)).reshape(self.raw_params['num_chans'],self.chan_size)
        block=block.astype(float)
        logger.info("Read block 1")

        for i in range(1,self.blocks_to_read):
            
            logger.info(f"Reading block {i+1} out of {self.blocks_to_read} ")

            nxt_block= (np.frombuffer(self.file_handler.read(self.block_read_size),offset=self.header_size
                          , dtype=np.int8)).reshape(self.raw_params['num_chans'],self.chan_size)
            nxt_block=nxt_block.astype(float)

            block=np.hstack((block, nxt_block))
        
        block_cmplx=block[: , ::2] + 1j*block[: , 1::2]
        
        return(block_cmplx)
        
        
    def write_blocks(self, data_chunk):
        """
        Function to write and save the dispersed block/s as GUPPI RAW file.

        Parameters
        ----------
        data_chunk : numpy.ndarray
            Dispersed complex series.

        Returns
        -------
        None.

        """
        
        logger.info(f"Writing block/s")
        
        self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)
        logger.info(f"Starting file handle position {self.file_handler.tell()} ")
        
        final_data=np.empty((self.data_size))

        for i in range(self.blocks_to_read):
            
            logger.info(f"Writing block {i+1} out of {self.blocks_to_read} ")
            
            self.file_handler.seek(self.header_size, 1)
            
            s=int(i*self.chan_size/2)
            e=int(i*self.chan_size/2 + self.chan_size/2)

            block_i=data_chunk[:, s:e]
            
            final_data[::2]=np.ravel(block_i.real)
            final_data[1::2]=np.ravel(block_i.imag)
            
            self.file_handler.write(np.array(final_data, dtype=np.int8).tobytes())
            
            logger.info(f"Ending file handle position {self.file_handler.tell()} ")
            logger.info(f"Writing complete")
     
        
    @classmethod
    def gauss(cls, x=None, x0=None, fwhm=None, a=None, width=None):
        """
        Create a Gaussian pulse profile according to passed parameters.

        Parameters
        ----------
        x : array, optional
            Time series. The default is None; (np.arange(width))
        x0 : float, optional
            index of peak value. The default is None; (width/2)
        fwhm : float, optional
            Full width at half maximum. The default is None; (width/2)
        a : float, optional
            Peak value of the gaussian profile. The default is None.
        width : int, optional
            Total span of the pulse to be generated. The default is None.

        Returns
        -------
        G : array
            Generated gaussian pulse profile

        """
        if x is None:
            x=np.arange(width)
        if x0 is None:
            x0=width/2
        if fwhm is None:
            fwhm=width/2
            
        sigma = (fwhm/2) / np.sqrt(2*np.log(2))
        
        if a is None:
            a= 1/(sigma*np.sqrt(2*np.pi))
        
        G= a  * np.exp(-(x-x0)**2 / (2*sigma**2))
        return G
        
    
    @classmethod
    def disperse_filterbank(cls, frame, params, b_type='N', save=True, op_dir=None):
        """
        Function to inject natural and artificial broadband signals in intensity domain.

        Parameters
        ----------
        frame : setigen.Frame
            A filterbank frame of data, from setigen.
        params : dict
            A dictionary with broadband parameters; width, snr(signal to noise ratio), t0(start time for injection),
            DM(dispersion measure), x(optional)(desired power law).
        b_type : str, optional
            inject any one of the 4 broadband signals;1 Natural signal and 3 Artificial signals (generated by flipping either time axis, frequency axis or both).
            The default is 'N'.
            'N': Broadband signal with natural dispersion
            'A1', 'A2', 'A3': Broadband signal with artificial dispersion (flipped time axis, frequency axis or both)
        save : bool, optional
            Save the frame with injected signal as a filterbank file . The default is True.
        op_dir : str, optional
            Full path to the output directory. Filename will be automatically appended (dispersed_frame.fil). Default is current working directory.

        Returns
        -------
        frame.data : setigen.Frame
            Frame of data with injected broadband signal

        """
        
        width, snr, t0, dm,x = params['width'], params['snr'], params['t0'], params['dm'], params.get('x',2)
        
        assert t0<frame.ts[-1], f"Start time {t0} seconds exceeds length of the file, {frame.ts[-1]} seconds"

        rms  = frame.get_intensity(snr=snr)
        fch1 = frame.get_frequency(frame.fchans-1)

        width_in_chans = width / frame.dt
        t0_in_samps = (t0 / frame.dt) - frame.ts[0]
    
        tdel_in_samps = 4.15e-3 * dm * ((frame.fs/1e9)**(-x) - (fch1/1e9)**(-x)) / frame.dt
        t0_in_samps = t0_in_samps + tdel_in_samps 

        t = np.arange(frame.tchans)
        t2d, tdel2d = np.meshgrid(t, t0_in_samps)
        profile = inj_broadband.gauss(t2d, tdel2d, width_in_chans, rms)
        
        if b_type=='N':
            pass
        if b_type=='A1':
            profile=np.flip(profile)
        if b_type=='A2':
            profile=np.flip(profile, axis=0)
        if b_type=='A3':
            profile=np.flip(profile, axis=1)

        frame.data +=profile.T

        plt.figure(figsize=(10,6))
        frame.plot()
        
        if save:
            if op_dir is None:
                loc= os.path.join(os.getcwd(), 'dispersed_frame.fil')
            else:
                loc= os.path.join(op_dir, 'dispersed_frame.fil')
                
            frame.save_fil(filename = loc)

        return(frame.data)
