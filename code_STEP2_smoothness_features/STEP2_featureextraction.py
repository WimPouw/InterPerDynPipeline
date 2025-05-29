import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from scipy import integrate
from scipy import interpolate
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import os
import glob

# specifically we might be interested in computing the smoothness of the distance
inputfol = '../dataoutput_STEP1_2_timeseries/'
outputfol = '../dataoutput_STEP2_features/'
metadata = pd.read_csv('../meta/project_pointsibling_metadata_starttimes.csv', encoding='latin1')
constantwindowsize_sec = 100 # we want an equal portion of each timeseries to be analyzed

# function to calculate spectral arc length
def spectral_arclength(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified spectral
    arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    It is suitable for movements that are a few seconds long, but for long
    movements it might be slow and results might not make sense (like any other
    smoothness metric).

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = spectral_arclength(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs/nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf/max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc)*1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th)*1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1]+1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

# function to calculate the dimensionless squared jerk metric from position data
def dimensionless_squared_jerk_from_position(ts, time):
    """
    Calculate the dimensionless squared jerk metric from position data.
    
    Parameters:
    -----------
    ts : array_like
        Position data points, should be a 1D numpy array
    time : array_like
        Time points corresponding to the ts data, should be a 1D numpy array
    
    Returns:
    --------
    float
        Dimensionless squared jerk metric or NaN if calculation fails
    """
    try:
        # First check the raw inputs
        ts = np.array(ts, dtype=float)
        time = np.array(time, dtype=float)
        
        print(f"Original shape - ts: {ts.shape}, time: {time.shape}")
        print(f"NaN count before processing - ts: {np.isnan(ts).sum()}, time: {np.isnan(time).sum()}")
        
        # Input validation before preprocessing
        if len(ts) != len(time):
            print(f"Error: Arrays must have the same length. ts: {len(ts)}, time: {len(time)}")
            return np.nan
            
        if len(ts) < 11:  # Minimum length for savgol filter
            print(f"Error: Input arrays too short for savgol window size=11. Length: {len(ts)}")
            return np.nan
        
        # Check if time has NaNs - we need to fix time first
        if np.isnan(time).any():
            print("Warning: Time array contains NaNs, filling with linear sequence")
            valid_time_mask = ~np.isnan(time)
            if not valid_time_mask.any():
                print("Error: All time values are NaN")
                return np.nan
                
            # Create a proper time sequence
            time = np.linspace(np.nanmin(time), np.nanmax(time), len(time))
        
        # Check if input data is all NaN
        if np.isnan(ts).all():
            print("Error: All input values are NaN")
            return np.nan
        
        # Identify valid data points for interpolation
        valid_mask = ~np.isnan(ts)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            print(f"Error: Not enough valid data points for interpolation. Found {len(valid_indices)}")
            return np.nan
        
        # Handle the interpolation more carefully
        # 1. Use valid points to interpolate
        if np.isnan(ts).any():
            try:               
                # Create interpolator with valid points only
                valid_time = time[valid_mask]
                valid_data = ts[valid_mask]
                
                # Sort by time to ensure proper interpolation
                sort_idx = np.argsort(valid_time)
                valid_time = valid_time[sort_idx]
                valid_data = valid_data[sort_idx]
                
                # Create interpolation function
                f = interpolate.interp1d(
                    valid_time, valid_data,
                    bounds_error=False,
                    fill_value=(valid_data[0], valid_data[-1])  # Extrapolate with edge values
                )
                
                # Apply interpolation
                ts_interpolated = f(time)
                
                # Check if interpolation fixed all NaNs
                if np.isnan(ts_interpolated).any():
                    print(f"Warning: Interpolation failed to fix all NaNs. Remaining: {np.isnan(ts_interpolated).sum()}")
                    # Last resort: replace any remaining NaNs with nearest valid value
                    ts_interpolated = pd.Series(ts_interpolated).fillna(method='ffill').fillna(method='bfill').values
                
                ts = ts_interpolated
            except Exception as e:
                print(f"Error during interpolation: {str(e)}")
                return np.nan
        
        # Verify no NaNs remain after preprocessing
        if np.isnan(ts).any() or np.isnan(time).any():
            print("Error: Failed to remove all NaN values")
            return np.nan
            
        # Ensure time steps are uniform for derivative calculation
        uniform_time = np.linspace(time[0], time[-1], len(time))
        if not np.allclose(time, uniform_time):
            # If time is not uniform, resample the data
            print("Warning: Time steps not uniform, resampling data")
            # Use scipy's interpolation again to resample
            f = interpolate.interp1d(time, ts, bounds_error=False, fill_value='extrapolate')
            ts = f(uniform_time)
            time = uniform_time
        
        # Calculate the time step
        dt = np.diff(time)[0]
        
        if dt <= 0:
            print(f"Error: Time steps must be positive. Got dt={dt}")
            return np.nan
        
        # Print state after preprocessing
        print(f"Data after preprocessing - Length: {len(ts)}")
        print(f"NaN check after preprocessing - ts: {np.isnan(ts).any()}, time: {np.isnan(time).any()}")
        print(f"Range - ts: {np.min(ts)} to {np.max(ts)}, time: {np.min(time)} to {np.max(time)}")
        
        # Calculate speed (exactly as in original)
        speed = np.gradient(ts, dt)

        # Smooth savgol filter (maintaining original settings)
        speed = savgol_filter(speed, 11, 3)

        # Calculate acceleration (exactly as in original)
        acceleration = np.gradient(speed, dt)

        # Smooth (maintaining original settings)
        acceleration = savgol_filter(acceleration, 11, 3)
        
        # Calculate jerk (exactly as in original)
        jerk = np.gradient(acceleration, dt)
        
        # Smooth (maintaining original settings)
        jerk = savgol_filter(jerk, 11, 3)
        
        # Calculate movement duration (D)
        movement_duration = time[-1] - time[0]
        
        if movement_duration <= 0:
            print(f"Error: Movement duration must be positive. Got {movement_duration}")
            return np.nan
        
        # Calculate movement amplitude by integrating speed
        position = integrate.simpson(speed, x=time)
        movement_amplitude = abs(position)  # Use absolute value to ensure positive
        
        # Prevent division by zero
        epsilon = 1e-10
        if movement_amplitude < epsilon:
            print(f"Warning: Movement amplitude very small ({movement_amplitude}). Using epsilon.")
            movement_amplitude = epsilon
            
        # Calculate the squared jerk
        squared_jerk = jerk ** 2
        
        # Integrate the squared jerk
        integrated_squared_jerk = integrate.simpson(squared_jerk, x=time)
        
        # Ensure positive value for integral of squared jerk
        if integrated_squared_jerk < 0:
            print(f"Warning: Negative integral of squared jerk: {integrated_squared_jerk}. Using absolute value.")
            integrated_squared_jerk = abs(integrated_squared_jerk)
        
        # Calculate the dimensionless squared jerk
        dimensionless_jerk = integrated_squared_jerk * (movement_duration**5 / movement_amplitude**2)
        
        # Final sanity check
        if np.isnan(dimensionless_jerk) or np.isinf(dimensionless_jerk):
            print(f"Warning: Result is {dimensionless_jerk}. Details: ")
            print(f"  Movement duration: {movement_duration}")
            print(f"  Movement amplitude: {movement_amplitude}")
            print(f"  Integrated squared jerk: {integrated_squared_jerk}")
            return np.nan
        # log the result
        
        print(f"Dimensionless squared jerk: {dimensionless_jerk}")
        # log the dimensionless jerk
        dimensionless_jerk = np.log(dimensionless_jerk)

        return dimensionless_jerk
        
    except Exception as e:
        print(f"Error calculating dimensionless squared jerk: {str(e)}")
        return np.nan
    
# df for smoothness date
newdfcolumns = ['videoID','timeadjusted', 'originalsamples','adjustedsamples', 'start_time_analysiswindow', 'end_time_analysiswindow', 'perc_twopersonsdetected', 'average_com_movementp1', 'average_com_movementp2', 'smoothness_distancecom', 'SPARC_smoothness_distancecom', 'smoothness_distancecentroid', 'smoothness_xy_average_com_p1', 'smoothness_xy_average_com_p2', 'smoothness_xy_average_centroid_p1', 'smoothness_xy_average_centroid_p2', 'smoothness_p1_proximity', 'smoothness_p2_proximity']
newdf = pd.DataFrame(columns=newdfcolumns)

# check for each csv file for layer2 data
layer2dat = glob.glob(inputfol + '*.csv')
#print(layer2dat)

# loop over the csv layer 2 data
for vids in layer2dat:
     print(vids)
     # Load the CSV timeseries file
     ts = pd.read_csv(vids)
     # get the features
     videoID = os.path.basename(vids).split('_processed_data_layer2.csv')[0]
     originalsamples = len(ts["time"])
     perc_twopersonsdetected = ts['both_tracked'].sum() / len(ts)
     # check metadata start
     start = metadata[metadata['VIDEO_ID'] == videoID]['start'].values
     print("start time of this video: " + str(start))
     # calculate the average movement of the com for each person
     average_com_movementp1 = np.mean(np.sqrt((ts['com_p1_x'].diff()**2 + ts['com_p1_y'].diff()**2)))
     average_com_movementp2 = np.mean(np.sqrt((ts['com_p2_x'].diff()**2 + ts['com_p2_y'].diff()**2)))
     # add a time adjusted variable to the dataset
     if start.size > 0: 
        # check if endtime not greater than the last time in the dataset
        if (start[0] + constantwindowsize_sec) > ts['time'].max():
            print("End time is greater than the last time in the dataset, setting end time to max value")
        timeadjusted = "TRUE - Adjusted to start at + " + str(start[0]) + "With a window length of: " + str(constantwindowsize_sec) + " seconds"
        # Take a timeseries chunk of 150 seconds
        ts = ts[(ts['time'] >= start[0]) & (ts['time'] <= start[0] + constantwindowsize_sec)]
     if start.size == 0:
        timeadjusted = "FALSE - Not adjusted to start time as no start time was given for this video, window length is still set to: " + str(constantwindowsize_sec) + " seconds"
        ts = ts[(ts['time'] <= constantwindowsize_sec)]
     adjustedsamples = len(ts["time"])
     print(timeadjusted) 
     # fps is mode timestaps per second 
     fps = 1/(ts['time'].diff().mode()[0])
     # add time start and time end
     start_time_analysiswindow = start[0] if start.size > 0 else 0
     end_time_analysiswindow = start_time_analysiswindow + constantwindowsize_sec
     # calculate the smoothness of the distance between the com and centroid
     smoothness_distancecom = dimensionless_squared_jerk_from_position(ts['distance_com'].values, ts['time'].values)
     # take the derivative of the distance
     distancecomspeed = np.gradient(ts['distance_com'].values, ts['time'].values)
     SPARCsmoothness_distancecom = spectral_arclength(distancecomspeed, 1/fps, padlevel=4, fc=10.0, amp_th=0.05)
     # it is the first value of the distancecomspeed
     SPARCsmoothness_distancecom = SPARCsmoothness_distancecom[0]
     #print(smoothness_distancecom)
     smoothness_distancecentroid = dimensionless_squared_jerk_from_position(ts['distance'].values, ts['time'].values)
     # calculate the smoothness of the xy positions for each person
     smoothness_xy_average_com_p1 = (dimensionless_squared_jerk_from_position(ts['com_p1_x'],ts['time'].values)+dimensionless_squared_jerk_from_position(ts['com_p1_y'],ts['time'].values))/2
     smoothness_xy_average_com_p2 = (dimensionless_squared_jerk_from_position(ts['com_p2_x'],ts['time'].values)+dimensionless_squared_jerk_from_position(ts['com_p2_y'],ts['time'].values))/2
     smoothness_xy_average_centroid_p1 = (dimensionless_squared_jerk_from_position(ts['centroid_p1_x'],ts['time'].values)+dimensionless_squared_jerk_from_position(ts['centroid_p1_y'],ts['time'].values))/2
     smoothness_xy_average_centroid_p2 = (dimensionless_squared_jerk_from_position(ts['centroid_p2_x'],ts['time'].values)+dimensionless_squared_jerk_from_position(ts['centroid_p2_y'],ts['time'].values))/2
     # calculate the smoothness of the proximity approach
     smoothness_p1_proximity = dimensionless_squared_jerk_from_position(ts['p1_com_approach_pos'].values, ts['time'].values)
     smoothness_p2_proximity = dimensionless_squared_jerk_from_position(ts['p2_com_approach_pos'].values, ts['time'].values)
     # append to the new df using concat
     newdf = pd.concat([newdf, pd.DataFrame([[videoID, timeadjusted, originalsamples, adjustedsamples, start_time_analysiswindow, end_time_analysiswindow, perc_twopersonsdetected, average_com_movementp1, average_com_movementp2, smoothness_distancecom, SPARCsmoothness_distancecom, smoothness_distancecentroid, smoothness_xy_average_com_p1, smoothness_xy_average_com_p2, smoothness_xy_average_centroid_p1, smoothness_xy_average_centroid_p2, smoothness_p1_proximity, smoothness_p2_proximity]], columns=newdfcolumns)], ignore_index=True)

# save the new df
newdf.to_csv(outputfol + 'smoothness_data.csv', index=False, encoding='latin1')
newdf.head()
# print done
print("Done with smoothness processing pipeline!")
