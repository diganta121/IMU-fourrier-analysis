import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import collections


SERIAL_PORT = 'COM6' 
BAUD_RATE = 115200

# Signal Processing Constants
FS = 100  # Sampling Rate
N_FFT = 256  # FFT Window Size

data_buffer = {
    'ax': collections.deque(np.zeros(N_FFT), maxlen=N_FFT),
    'az': collections.deque(np.zeros(N_FFT), maxlen=N_FFT),
    'gz': collections.deque(np.zeros(N_FFT), maxlen=N_FFT),
}

# Calculated frequency labels for the FFT plot (0 to FS/2)
freq_labels = np.fft.fftfreq(N_FFT, d=1/FS)[:N_FFT // 2]

# Attempt to initialize Serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Successfully connected to {SERIAL_PORT}")
    SERIAL_CONNECTED = True

except serial.SerialException as e:
    print(f"ERROR: Could not open serial port {SERIAL_PORT}. Running in SIMULATION mode.")
    print("Please check connection and port name if you want real data.")
    print(f"Reason: {e}")
    SERIAL_CONNECTED = False
    
    sim_time = 0
    sim_start_time = time.time()


# --- Data Reading and Processing Functions ---

def read_serial_data():
    """
    Reads one line of IMU data from the serial port.
    The expected format from ESP32 is a comma-separated string like:
    'ax_val,ay_val,az_val,gx_val,gy_val,gz_val\n'
    """
    global sim_time
    
    if SERIAL_CONNECTED:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Basic parsing: expects 6 float values separated by commas
                values = [float(v) for v in line.split(',')[:6]]
                if len(values) == 6:
                    return {
                        'ax': values[0], 'ay': values[1], 'az': values[2],
                        'gx': values[3], 'gy': values[4], 'gz': values[5]
                    }
        except ValueError:
            return None
        except serial.SerialTimeoutException:
            return None
    
    # --- SIMULATION MODE (Fallback if serial is not connected) ---
    else:
        # Simulate a clear 2 Hz vibration on the Z-axes
        sim_time = time.time() - sim_start_time
        az_sim = 9.81 + 1.0 * np.sin(2 * np.pi * 2 * sim_time) + (np.random.rand() - 0.5) * 0.1
        gz_sim = 20.0 * np.cos(2 * np.pi * 2 * sim_time) + (np.random.rand() - 0.5) * 0.5
        
        # Other axes are random noise
        return {
            'ax': (np.random.rand() - 0.5) * 0.1, 'ay': (np.random.rand() - 0.5) * 0.1, 'az': az_sim,
            'gx': (np.random.rand() - 0.5) * 0.05, 'gy': (np.random.rand() - 0.5) * 0.05, 'gz': gz_sim
        }
    return None

def calculate_fft_magnitude(data):
    """Calculates the FFT and returns the single-sided magnitude spectrum."""
    # 1. Apply a window function (e.g., Hamming) to reduce spectral leakage
    windowed_data = data * np.hamming(N_FFT)
    
    # 2. Perform the FFT
    fft_result = np.fft.fft(windowed_data)
    
    # 3. Calculate magnitude and take the single-sided spectrum (0 to FS/2)
    # We ignore the first point (DC offset, which is the mean) for clarity in vibration analysis
    magnitude = np.abs(fft_result)
    single_sided_magnitude = magnitude[1:N_FFT // 2]
    
    # 4. Scale by 2/N_FFT (except DC and Nyquist, but simplified to N_FFT for plotting)
    return single_sided_magnitude / (N_FFT / 2)


# --- Plotting Setup ---

# Create the figure and subplots
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Real-Time IMU Data and FFT Analysis', fontsize=16)
plt.style.use('dark_background')

# --- Time Domain Plot (ax_time) ---
line_az, = ax_time.plot(np.arange(N_FFT), data_buffer['az'], label='Accel Z (m/s²)', color='#3b82f6')
line_gz, = ax_time.plot(np.arange(N_FFT), data_buffer['gz'], label='Gyro Z (rad/s)', color='#10b981')
ax_time.set_title('Time Domain Signal (Last {} Samples)'.format(N_FFT))
ax_time.set_xlabel('Sample Index')
ax_time.set_ylabel('Amplitude')
ax_time.legend(loc='upper right')
ax_time.grid(linestyle='--', alpha=0.5)
ax_time.set_ylim(-15, 15) # Example range, adjust if needed

# --- Frequency Domain Plot (ax_freq) ---
# We use the frequency labels, skipping the DC component (index 0)
bar_width = (FS / N_FFT) * 0.8
bars_az = ax_freq.bar(freq_labels[1:], calculate_fft_magnitude(data_buffer['az']), width=bar_width, color='#facc15', alpha=0.7, label='Accel Z |FFT|')
bars_gz = ax_freq.bar(freq_labels[1:] + bar_width, calculate_fft_magnitude(data_buffer['gz']), width=bar_width, color='#f43f5e', alpha=0.7, label='Gyro Z |FFT|')

ax_freq.set_title(f'Frequency Domain Magnitude Spectrum (FS={FS}Hz, $\Delta f$={(FS/N_FFT):.2f}Hz)')
ax_freq.set_xlabel('Frequency (Hz)')
ax_freq.set_ylabel('Magnitude')
ax_freq.set_xlim(0, FS / 2)
ax_freq.set_ylim(0, 5) # Adjust max magnitude based on expected values
ax_freq.legend(loc='upper right')
ax_freq.grid(linestyle='--', alpha=0.5)


# --- Animation Update Function ---

def update_plot(frame):
    """Function called repeatedly by FuncAnimation to update the plot."""
    # 1. Read new data point
    new_data = read_serial_data()
    
    if new_data is not None:
        # 2. Update data buffers (efficiently done by deque)
        data_buffer['az'].append(new_data['az'])
        data_buffer['gz'].append(new_data['gz'])
        
        # 3. Time Domain Update
        line_az.set_ydata(data_buffer['az'])
        line_gz.set_ydata(data_buffer['gz'])
        
        # 4. FFT Calculation and Frequency Domain Update
        fft_az = calculate_fft_magnitude(np.array(data_buffer['az']))
        fft_gz = calculate_fft_magnitude(np.array(data_buffer['gz']))
        
        # Update Accel Z bars
        for bar, mag in zip(bars_az, fft_az):
            bar.set_height(mag)

        # Update Gyro Z bars
        for bar, mag in zip(bars_gz, fft_gz):
            bar.set_height(mag)
            
    # Return the artist objects that were modified
    return line_az, line_gz, *bars_az, *bars_gz

# --- Main Loop ---

# Set the interval to match the sampling rate (1000 ms / FS samples)
interval_ms = 1000 / FS

# Create the animation object
ani = animation.FuncAnimation(fig, update_plot, interval=interval_ms, blit=True, cache_frame_data=False)

try:
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    plt.show()
finally:
    # Clean up the serial connection when the plot window is closed
    if SERIAL_CONNECTED and ser.is_open:
        ser.close()
        print("Serial connection closed.")
