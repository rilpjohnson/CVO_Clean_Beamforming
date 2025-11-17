import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime
import matplotlib.dates as mdates
from obspy.clients.fdsn import Client
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
import pickle, time
import sys
import os
from matplotlib.colors import to_hex

import cleanbf

#===== EVENT CONFIGURATION (PARAMETERIZED FOR ANY EVENT) =====
# Define your event here by modifying these parameters:
EVENT_CONFIG = {
    'name': 'aftershock',                          # name for output files
    'network': 'CC',                               # network code
    'station': 'MILD',                             # station code
    'location': '*',                               # location code
    'channel': '*DF',                              # channel code
    'event_time': obspy.UTCDateTime('2023-01-16T00:15:06'),  # event origin time
    'window_start': obspy.UTCDateTime('2023-01-16T00:15:00.600000Z'),  # analysis window start
    'window_end': obspy.UTCDateTime('2023-01-16T00:40:00.600000Z'),    # analysis window end
    'pkl_file': 'data/pkl/clean_aftershock_3.pkl', # cached beamforming result
    'slowness_threshold': 2,                       # slowness threshold (s/km)
    'power_threshold_frac': 0.15,                  # power threshold (fraction of max)
    'freq_min': 1,                                 # min frequency (Hz)
    'freq_max': 15,                                # max frequency (Hz)
}

# Extract config for readability
name = EVENT_CONFIG['name']
t1 = EVENT_CONFIG['window_start']
t2 = EVENT_CONFIG['window_end']
event = EVENT_CONFIG['event_time']
pkl_file3 = EVENT_CONFIG['pkl_file']
slowness_threshold = EVENT_CONFIG['slowness_threshold']
power_thresh_frac = EVENT_CONFIG['power_threshold_frac']
freq_min = EVENT_CONFIG['freq_min']
freq_max = EVENT_CONFIG['freq_max']

client = Client("IRIS")

# Define parameters for the inventory request
network = EVENT_CONFIG['network']
station = EVENT_CONFIG['station']
location = EVENT_CONFIG['location']
channel = EVENT_CONFIG['channel']
starttime = t1
endtime = t2
inventory_path = "demos/inventory.xml" # change to 



try:
    # Fetch inventory
    inventory = client.get_stations(network=network,
                                    station=station,
                                    location=location,
                                    channel=channel,
                                    starttime=starttime,
                                    endtime=endtime,
                                    level="response")

    # Save as StationXML (IRIS XML format)
    inventory.write("demos/inventory.xml", format="STATIONXML")
    print("Inventory saved to inventory.xml")
except Exception as e:
    print(f"Error fetching inventory: {e}")
#for stn in stationList:
eq_stream = Client("IRIS").get_waveforms(network=network, location=location, station=station, channel=channel, starttime=starttime, endtime=endtime)
for t in eq_stream.traces:
        #t.plot()
        t.data = t.data.astype("int32")

inv = obspy.read_inventory('demos/inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream


#%%
## Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)
## define slowness grid to search
s_list = np.arange(-4, 4.01, 0.1)

## plot parameters
slowness_threshold = EVENT_CONFIG['slowness_threshold']
xticks = np.arange(0, 70.1, 10)
baz_ticks = [-180, -90, 0, 90, 180]

## loop parameters
loop_start = t1
loop_end = t2
loop_step = 1
loop_width = 2
phi = 0.1
win_length_sec = 1
freq_min = EVENT_CONFIG['freq_min']
freq_max = EVENT_CONFIG['freq_max']
separate_freqs = 0
freq_bin_width = 1



## plot an inset of 3 traces after amplitude drops off, blown up

#%% calculate 3-station clean result

calculate_new_beamform_result_3 = False # set to False after running the calculation once to save time

if calculate_new_beamform_result_3:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream.slice(loop_start, loop_end), loop_width = loop_width, loop_step = loop_step, 
                              verbose = False, phi = phi, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, # formerly freq 3-25
                              sxList = s_list, syList = s_list, prewhiten = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file3, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file3, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)
output_3 = output        
#%% Run obspy's traditional beamformer for runtime comparison (not needed for making figures)
if False:
    analysis_start_time = time.time()
    array_proc_output=obspy.signal.array_analysis.array_processing(eq_stream_3.slice(loop_start, loop_end), loop_width, loop_step/loop_width, 
                                                 s_list.min(), s_list.max(), s_list.min(), s_list.max(), np.diff(s_list)[0],
                                                 0, 0, freq_min, freq_max, loop_start, loop_end, prewhiten = False)
    print( time.time() - analysis_start_time) # about 0.6 seconds, compared to 58 seconds for sub-array CLEAN 


#%% plot 3-station clean result
sh = output['sh']
# normalize backazimuth into [-180, 180) for plotting (library uses 0..360 for back_az)
baz = ((output['back_az'] + 180) % 360) - 180
spec_4 = output['clean_polar_back']
r = 1e-5
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:, ind]
ac_max = np.quantile(spec_baz, 1)

# --- Enhanced plots: sharp backazimuth, slowness, power, and map (Cartopy) ---
# Build selection of times where slowness indicates a coherent arrival
w = sh > slowness_threshold
t_rel_all = output['t'] - (event - loop_start)
baz_all = ((output['back_az'] + 180) % 360) - 180
sh_all = sh

# spec_baz was computed for w earlier in the script; reuse if available
try:
    power_by_time = spec_baz.max(axis=0)
except Exception:
    # fallback: use sum over polar power (if available) or zeros
    try:
        power_by_time = np.einsum('hijk->hj', spec_4[:,:,:,w]).max(axis=0)
    except Exception:
        power_by_time = np.zeros(np.count_nonzero(w))

# === NEW APPROACH: Extract max power in slowness-backazimuth grid per time ===
# spec_4 shape: (1499, 18, 36, 41) = (time, slowness_y, backazimuth, slowness_magnitude)
# For each time, find (slowness, backazimuth) pair with max power
t_max_power = []
t_max_baz = []
t_max_sh = []
power_list = []

for h in range(spec_4.shape[0]):  # loop over time
    spec_t = spec_4[h, :, :, :]  # shape: (18, 36, 41)
    max_idx = np.unravel_index(np.argmax(spec_t), spec_t.shape)  # (sy_idx, baz_idx, sh_idx)
    max_power = spec_t[max_idx]
    
    sy_idx, baz_idx, sh_idx = max_idx
    # backazimuth from grid index: map [0,36) to [0, 360)
    baz_grid_val = (baz_idx / 36.0) * 360.0
    # slowness magnitude from grid index  
    sh_grid_val = output['sh'][sh_idx]
    
    t_max_power.append(h)
    power_list.append(max_power)
    t_max_baz.append(baz_grid_val)
    t_max_sh.append(sh_grid_val)

t_max_power = np.array(t_max_power)
power_list = np.array(power_list)
t_max_baz = np.array(t_max_baz)
t_max_sh = np.array(t_max_sh)

# select times with significant power (tunable threshold)
power_thresh = power_thresh_frac * power_list.max() if power_list.max() > 0 else 0
tmask = power_list > power_thresh

# Also restrict to coherent time window (find continuous high-power region)
# Find times where power > 1% of max and group them
power_window_mask = power_list > (0.01 * power_list.max())
if power_window_mask.sum() > 0:
    t_window_idx = np.where(power_window_mask)[0]
    t_window_start = max(0, t_window_idx[0] - 10)  # buffer 10 samples
    t_window_end = min(len(t_rel_all) - 1, t_window_idx[-1] + 10)
    window_mask = (t_max_power >= t_window_start) & (t_max_power <= t_window_end)
    tmask = tmask & window_mask

t_sel = t_rel_all[tmask]
baz_sel = t_max_baz[tmask]
sh_sel = t_max_sh[tmask]
power_sel = power_list[tmask]

# Set plot window bounds to coherent region
plot_t_min = t_sel.min() - 5 if len(t_sel) > 0 else t_rel_all.min()
plot_t_max = t_sel.max() + 5 if len(t_sel) > 0 else t_rel_all.max()

print(f'Selected {tmask.sum()} times out of {len(power_list)} with power > {power_thresh_frac*100:.0f}% of max')

# normalize power for plotting
if power_sel.max() > 0:
    pnorm = power_sel / power_sel.max()
else:
    pnorm = power_sel

# create a refined 2x2 figure: slowness, backazimuth (sharp), power, map
fig2 = plt.figure(figsize=(12, 8))
ax1 = fig2.add_subplot(2,2,1)
sc1 = ax1.scatter(t_sel, sh_sel, c=pnorm, cmap='viridis', s=50, edgecolor='k')
ax1.set_ylabel('Slowness (s/km)')
ax1.set_xlabel('Time (s after event)')
ax1.set_title('Slowness vs Time (Coherent Window)')
ax1.axhline(slowness_threshold, color='k', ls='--', lw=0.8)
ax1.set_xlim(plot_t_min, plot_t_max)
cax1 = fig2.add_axes([0.47, 0.55, 0.015, 0.32])
cb1 = fig2.colorbar(sc1, cax=cax1)
cb1.set_label('Normalized power')

ax2 = fig2.add_subplot(2,2,2)
sc2 = ax2.scatter(t_sel, baz_sel, c=pnorm, cmap='magma', s=40, edgecolor='k')
ax2.set_ylabel('Backazimuth (deg)')
ax2.set_xlabel('Time (s after event)')
ax2.set_title('Backazimuth vs Time (Coherent Window)')
ax2.set_yticks([0, 45, 90, 135, 180, 225, 270, 315])
ax2.set_xlim(plot_t_min, plot_t_max)

# smooth the backazimuth track by averaging unit vectors (handles circular wrap at 0/360)
if len(baz_sel) >= 3:
    # Convert to unit vectors in complex plane (0° = North, increases clockwise)
    theta = np.radians(baz_sel)
    v = np.exp(1j * theta)
    # Apply moving average in complex plane
    win = 5 if len(v) >= 5 else len(v)
    if win % 2 == 0:
        win -= 1  # ensure odd window
    kernel = np.ones(win) / win
    v_smooth = np.convolve(v, kernel, mode='same')
    # Convert back to degrees, keeping 0-360 range
    baz_smooth = (np.degrees(np.angle(v_smooth)) % 360)
    ax2.plot(t_sel, baz_smooth, color='k', lw=2.0, label='Smoothed', zorder=5)
    ax2.legend(loc='upper right')

cax2 = fig2.add_axes([0.95, 0.55, 0.015, 0.32])
cb2 = fig2.colorbar(sc2, cax=cax2)
cb2.set_label('Normalized power')

ax3 = fig2.add_subplot(2,2,3)
ax3.plot(t_sel, power_sel, color='C1', lw=1.2)
ax3.set_xlabel('Time (s after event)')
ax3.set_ylabel('Power (arb)')
ax3.set_title('Max Polar Power vs Time')
ax3.set_xlim(plot_t_min, plot_t_max)
ax3.axvline(0, color='gray', lw=0.8, ls=':')

# Polar backazimuth rose in 2x2 figure (bottom-right)
ax4 = fig2.add_subplot(2,2,4, projection='polar')
try:
    # Histogram of backazimuth weighted by power
    theta = np.radians(baz_sel)
    nbins = 36
    bins = np.linspace(0, 2*np.pi, nbins + 1)
    hist, edges = np.histogram(theta, bins=bins, weights=power_sel)
    # Normalize to fraction of total power
    if hist.sum() > 0:
        hist_frac = hist / hist.sum()
    else:
        hist_frac = hist
    width = edges[1] - edges[0]
    cmap = plt.get_cmap('magma')
    colors = cmap(hist_frac / (hist_frac.max() if hist_frac.max()>0 else 1.0))
    ax4.bar(edges[:-1], hist_frac, width=width, bottom=0.0, align='edge', color=colors, edgecolor='k', linewidth=0.5)
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)  # clockwise
    ax4.set_title('Backazimuth Rose\n(power fraction)', fontsize=10, pad=15)
except Exception as e:
    ax4.text(0.5, 0.5, f'Rose failed: {str(e)[:20]}', ha='center', va='center',
             transform=ax4.transAxes, fontsize=9)

# Save the polished 2x2 figure
outname = f'Fig4_{name}_2x2.png'
fig2.savefig(outname, dpi=300)
print(f'Saved polished figure to {outname}')
plt.close(fig2)


#%% ===== HOW TO USE FOR NEW EVENTS =====
# To process a different event, modify EVENT_CONFIG at the top of this script:
#
# EVENT_CONFIG = {
#     'name': 'event_name',                         # Name for output files (event_name_2x2.png, etc)
#     'network': 'XX',                              # FDSN network code
#     'station': 'YYYY',                            # Station code  
#     'location': '*',                              # Location code
#     'channel': '*DF',                             # Channel code
#     'event_time': obspy.UTCDateTime('YYYY-MM-DDTHH:MM:SS.FFFFFFZ'),  # Origin time
#     'window_start': obspy.UTCDateTime('YYYY-MM-DDTHH:MM:SS.FFFFFFZ'), # Analysis window start
#     'window_end': obspy.UTCDateTime('YYYY-MM-DDTHH:MM:SS.FFFFFFZ'),   # Analysis window end
#     'pkl_file': 'path/to/cached/beamform.pkl',   # Cached CLEAN beamforming result
#     'slowness_threshold': 2,                      # Min slowness (s/km) for coherence
#     'power_threshold_frac': 0.15,                 # Power threshold as fraction of max
#     'freq_min': 1,                                # Min frequency (Hz)
#     'freq_max': 15,                               # Max frequency (Hz)
# }
#
# Then run: python demos/copilot_help.py
# Output files: Fig4_<name>_2x2.png, Fig4_<name>_baz_polar.png, Fig4_<name>_map.html

#%% save figure (legacy)
plt.savefig('Fig4_Aftershock_20_3_paper.png', dpi = 300)

#%% plot 3-station results
plt.subplot(7,1,4)
times = mdates.date2num(eq_stream[0].times("utcdatetime"))

plt.plot(times, eq_stream[0].data)

sh = output['sh']
baz = output['back_az']-180
spec_4 = output['clean_polar_back']
r = 1e-2
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:,ind]
ac_max = np.quantile(spec_baz, 0.999)

#output['t']-(event - loop_start)
def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def image_trim_log(x): return(np.log(trim(x, ac_max, ac_max*r)))
def image_show_outliers(x):
    mean = np.einsum('ij->j', x)/x.shape[0]
    std = np.sqrt(np.einsum('ij->j', (x - mean)**2)/x.shape[0])
    x[x < (mean +2*std)] = 0
    return np.log(trim(x, ac_max, r*ac_max))

plt.subplot(7,1,5)
cleanbf.image(image_trim_log(spec_s), output['t'], sh, crosshairs = False)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks([0,1,2,3])
plt.title('e. Power vs. Time, Slowness (3 sensors)', loc = 'left', fontsize = 'small')
w = output['original_sh'] < 4

plt.subplot(7,1,6)
cleanbf.image(image_trim_log(spec_baz), output['t'], baz, crosshairs = False)
plt.ylabel('degrees')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('f. Power vs. Time, Backazimuth (Slowness > 2 s/km)', loc = 'left', fontsize = 'small')

plt.subplot(7,1,7)
im=cleanbf.image(image_show_outliers(spec_baz), output['t'], baz, crosshairs = False)
plt.xlabel('Time after earthquake (seconds)')
plt.ylabel('degrees')
# only plot x ticks for bottom panel to save space
plt.xticks(np.arange(0, 3001, 500))
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('g. Power vs. Time, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 'small')


#%%
fig = plt.gcf()
fig.set_size_inches(6.5, 9, forward=True) # max fig size for villarrica screen: 12.94x6.65 inch
fig.tight_layout()
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(im,cax=cbar_ax, label='Infrasound Power', ticks = [])
fig.subplots_adjust(right=0.91)

plt.savefig('Fig4_Aftershock_20_4_paper.png', dpi = 300)