import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import json


class BeamformingSimulator:
    def __init__(self):
        # Initial parameters (default)
        self.num_elements = 16
        self.frequency = 1e6  # Hz
        self.steering_angle = 0  # degrees
        self.element_spacing = 0.05  # m
        self.is_linear = True
        self.speed = 343  # Default speed of sound in air (m/s)
        self.colorbar_max = 0
        # Derived parameters
        self.update_derived_parameters()

        self.load_scenario('D:/1Programming/DSP/Task4/5G.json')

        # Setup visualization
        self.setup_plot()
        self.update(None)
        plt.show()

    def load_scenario(self, filename):
        print(f"Loading scenario: {filename}")
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            print(f"Parameters loaded: {params}")

            # Validate and update parameters
            self.frequency = params.get('frequency', self.frequency)
            self.num_elements = params.get('num_elements', self.num_elements)
            self.element_spacing = params.get('element_spacing', self.element_spacing)
            self.steering_angle = params.get('steering_angle', 0)  # Default to 0 if not specified
            self.is_linear = params.get('is_linear', True)
            self.speed = params.get('propagation_speed', self.speed)

            self.update_derived_parameters()

            # Update sliders and radio button states
            if self.s_freq:
                self.s_freq.set_val(self.frequency)
            if self.s_elements:
                self.s_elements.set_val(self.num_elements)
            if self.s_spacing:
                self.s_spacing.set_val(self.element_spacing)
            if self.s_angle:
                self.s_angle.set_val(self.steering_angle)
            if self.radio:
                self.radio.set_active(0 if self.is_linear else 1)

        except Exception as e:
            print(f"Error loading scenario: {e}")

    def update_derived_parameters(self):
        self.wavelength = self.speed / self.frequency
        self.k = 2 * np.pi / self.wavelength

    # def create_scenario_selector(self):
    #     ax_scenario = plt.axes([0.02, 0.75, 0.10, 0.15])
    #     self.radio_scenario = RadioButtons(ax_scenario, ('5G'), active=0)
    #     self.radio_scenario.on_clicked(self.handle_scenario_change)

    def handle_scenario_change(self, label):
        print(f"Scenario changed to: {label}")
        if label == '5G':
            self.load_scenario('D:/1Programming/DSP/Task4/5G.json')

    def calculate_array_factor(self, theta):
        positions = self.get_element_positions()
        AF = np.zeros_like(theta, dtype=complex)
 
        for pos in positions:
            if self.is_linear:
                phase = self.k * pos[0] * (np.sin(np.deg2rad(self.steering_angle)) - np.sin(theta))
            else:
                phase = self.k * (pos[0] * np.cos(theta) + pos[1] * np.sin(theta))
                phase -= self.k * (pos[0] * np.cos(np.deg2rad(self.steering_angle)) +
                                   pos[1] * np.sin(np.deg2rad(self.steering_angle)))
            AF += np.exp(1j * phase)

        return np.abs(AF) / self.num_elements

    def calculate_pressure_field(self, x, y):
        pressure = np.zeros((len(y), len(x)), dtype=complex)
        positions = self.get_element_positions()
        X, Y = np.meshgrid(x, y)
        steering_rad = np.deg2rad(self.steering_angle)
        if self.is_linear:
            steering_delays = positions[:, 0] * np.sin(steering_rad)
        else:
            steering_delays = (positions[:, 0] * np.cos(steering_rad) +
                               positions[:, 1] * np.sin(steering_rad))

        for pos, delay in zip(positions, steering_delays):
            r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
            phase = self.k * (r - delay)
            r0 = self.wavelength
            amplitude = r0 / (r + r0)
            pressure += amplitude * np.exp(1j * phase)

        pressure = np.abs(pressure) / self.num_elements
        if self.is_linear:
            return np.flip(pressure.T, (0, 1))
        else:
            return np.flip(pressure, (0, 1)) 

    def get_element_positions(self):
        if self.is_linear:
            x = np.linspace(-(self.num_elements - 1) / 2 * self.element_spacing,
                            (self.num_elements - 1) / 2 * self.element_spacing,
                            self.num_elements)
            return np.column_stack((x, np.zeros_like(x)))
        else:
            arc_length = self.element_spacing * (self.num_elements - 1)
            radius = arc_length / (2 * np.pi)
            angles = np.linspace(0, np.pi, self.num_elements)
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            return np.column_stack((x, -y))

    def setup_plot(self):
        self.fig = plt.figure(figsize=(15, 5))
        self.ax_polar = self.fig.add_subplot(131, projection='polar')
        self.ax_elements = self.fig.add_subplot(132)
        self.ax_field = self.fig.add_subplot(133)
        plt.subplots_adjust(bottom=0.3)
        self.create_sliders()
        ax_radio = plt.axes([0.02, 0.35, 0.1, 0.1])
        self.radio = RadioButtons(ax_radio, ('Linear', 'Curved'))
        self.radio.on_clicked(self.update)
        ax_radio_mode = plt.axes([0.02, 0.45, 0.1, 0.1])  # Adjust the position
        self.radio_mode = RadioButtons(ax_radio_mode, ('5G', 'Ultrasound', 'Tumor Ablation'), active=0)  # '5G' selected by default
        self.radio_mode.on_clicked(self.handle_mode_change)
        # self.create_scenario_selector()

        # Define the handle_mode_change method
    def handle_mode_c  hange(self, label):
        if label == 'Ultrasound':
            self.frequency = 0.5e10  # 0.500 x 10^10 Hz
            self.num_elements = 6
            self.s_freq.set_val(self.frequency)  # Update the frequency slider
            self.s_elements.set_val(self.num_elements)  # Update the elements slider
        elif label == 'Tumor Ablation':
            self.frequency = 0.5e10  # 0.500 x 10^10 Hz
            self.num_elements = 6
            self.s_freq.set_val(self.frequency)  # Update the frequency slider
            self.s_elements.set_val(self.num_elements)  # Update the elements slider
        elif label == '5G':
            self.load_scenario('D:/1Programming/DSP/Task4/5G.json')  # Reload 5G settings

    def create_sliders(self):
        ax_angle = plt.axes([0.1, 0.15, 0.65, 0.02])
        self.s_angle = Slider(ax_angle, 'Steering Angle (°)',
                              -90, 90, valinit=self.steering_angle)

        ax_elements = plt.axes([0.1, 0.1, 0.65, 0.02])
        self.s_elements = Slider(ax_elements, 'Number of Elements',
                                 4, 32, valinit=self.num_elements, valstep=1)

        ax_freq = plt.axes([0.1, 0.05, 0.65, 0.02])
        self.s_freq = Slider(ax_freq, 'Frequency (Hz)',
                             0.5e6, 30e9, valinit=self.frequency)

        ax_spacing = plt.axes([0.1, 0.2, 0.65, 0.02])
        self.s_spacing = Slider(ax_spacing, 'Element Spacing (m)',
                                0.01, 0.1, valinit=self.element_spacing)

        for slider in [self.s_angle, self.s_elements, self.s_freq, self.s_spacing]:
            slider.on_changed(self.update)

    def update(self, _):
        self.steering_angle = self.s_angle.val
        self.num_elements = int(self.s_elements.val)
        self.frequency = self.s_freq.val
        if not (0.5e6 <= self.frequency <= 30e9):
            print("Frequency out of bounds! Resetting to default.")
            self.frequency = 1e6
            self.s_freq.set_val(self.frequency)
        self.wavelength = self.speed / self.frequency
        self.k = 2 * np.pi / self.wavelength
        self.element_spacing = self.s_spacing.val
        self.is_linear = self.radio.value_selected == 'Linear'
        self.ax_polar.clear()
        self.ax_elements.clear()
        self.ax_field.clear()

        theta = np.linspace(-np.pi, np.pi, 1000)
        af = self.calculate_array_factor(theta)
        af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-6))
        af_db -= np.max(af_db)
        self.ax_polar.plot(theta, af_db)
        self.ax_polar.set_rmin(-40)
        self.ax_polar.set_title('Far-field Pattern (dB)')

        positions = self.get_element_positions()
        self.ax_elements.scatter(positions[:, 0], positions[:, 1], c='r')
        self.ax_elements.set_title('Array Geometry')
        self.ax_elements.set_aspect('equal')
        self.ax_elements.grid(True)
        max_dim = np.max(np.abs(positions)) * 1.2
        self.ax_elements.set_xlim(-max_dim, max_dim)
        self.ax_elements.set_ylim(-max_dim, max_dim)

        if self.is_linear:
            steering_rad = np.deg2rad(self.steering_angle)
            self.ax_elements.arrow(0, 0,
                                   max_dim / 2 * np.sin(steering_rad),
                                   max_dim / 2 * np.cos(steering_rad),
                                   head_width=max_dim / 20, color='b', alpha=0.5)

        field_size = max_dim * 10
        x = np.linspace(-field_size / 2, field_size / 2, 200)
        y = np.linspace(-field_size / 2, field_size / 2, 200)
        pressure = self.calculate_pressure_field(x, y)

        pressure_db = 20 * np.log10(np.maximum(pressure, 1e-6))
        vmin = -100
        vmax = 0
        im = self.ax_field.pcolormesh(x, y, pressure_db,
                                      vmin=vmin, vmax=vmax,
                                      cmap='jet', shading='auto')
        if self.colorbar_max == 0:
            cbar = self.fig.colorbar(im, ax=self.ax_field)
            cbar.set_label('Relative Pressure (dB)')
            cbar.set_ticks(np.linspace(vmin, vmax, 11))
            cbar.set_ticklabels(np.linspace(0, 100, 11).astype(int))
            self.colorbar_max = 1
        self.ax_field.set_title('Interference Field')
        self.ax_field.set_aspect('equal')
        self.ax_elements.set_xlabel('Position (m)')
        self.ax_elements.set_ylabel('Position (m)')
        self.ax_field.set_xlabel('Position (m)')
        self.ax_field.set_ylabel('Position (m)')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    simulator = BeamformingSimulator()