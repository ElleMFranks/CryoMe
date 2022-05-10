import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib import ticker
from numpy.core.multiarray import zeros

import data_structs


# region Plot utility objects/functions.
@dataclass
class PlotColours:
    dark: str
    light: str
    blue: str
    green: str
    orange: str


@dataclass()
class PlotVars:
    label_font_size: int
    pixel_line_width: float
    font_colour: str
    colours: PlotColours


@dataclass()
class AxisMinMax:
    mini: float
    maxi: float


@dataclass()
class XYAxisVars:
    x_label: str
    y_label: str
    x_minmax: AxisMinMax
    y_minmax: AxisMinMax


@dataclass()
class CartesianDataset:
    x_data: np.array
    y_data: np.array
    label: str


@dataclass()
class PlotInstDetails:
    stage_ut: np.array
    lna_ut: np.array

def _get_std_fig(color: str = '#181818') -> plt.Figure:
    """Returns a standard figure setup. Default colour is dark."""
    fig = plt.figure()
    # region Set figure variables.
    #fig.set_size_inches(20.92, 11.77)
    #fig.set_dpi(91.79)
    fig.tight_layout(pad=5)
    fig.set_facecolor(color)
    # endregion
    return fig
# endregion
   

# region Cartesian plots
class CartesianPlot:
    """Base class for cartesian plots."""
    colors = PlotColours(
        '#181818', '#eeeeee', '#030bfc', '#057510', '#f74600')
    pixel_line_width = 0.7
    label_font_size = 12
    font_color = colors.light

    def __init__(self, axis: plt.Axes, title: str) -> None:
        """Constructor for the CartesianPlot class.
        
        Args:
            axis: The axis to set up.
            title: The title of the plot.
        """
        
        self.axis = axis
        self.plot_labels = []
        self.plots = []
        plt.tight_layout()

        # region Set axis variables.
        self.axis.set_title(title, color=self.font_color, 
                            fontsize=self.label_font_size + 2)
        self.axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        self.axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        self.axis.grid(linestyle='--', which='major', 
                       linewidth=self.pixel_line_width + 0.3)
        self.axis.grid(linestyle='--', which='minor', 
                       linewidth=self.pixel_line_width - 0.3)
        self.axis.tick_params('both', colors=self.font_color,
                              labelsize=self.label_font_size)
        # endregion

    def set_axis(self, axis_vars: XYAxisVars) -> None:
        """Set limits and labels for x and y axis."""
        self.axis.set_ylim(axis_vars.y_minmax.mini, axis_vars.y_minmax.maxi)
        self.axis.set_xlim(axis_vars.x_minmax.mini, axis_vars.x_minmax.maxi)
        self.axis.set_xlabel(axis_vars.x_label, color=self.font_color, 
                             fontsize=self.label_font_size)
        self.axis.set_ylabel(axis_vars.y_label, color=self.font_color, 
                             fontsize=self.label_font_size)

    def add_legend(self) -> None:
        """Adds a legend to the plot with all the datasets/labels."""
        labels = (plot.get_label() for plot in self.plots)
        self.axis.legend(
            self.plots, labels, loc='lower left', numpoints=1, 
            fontsize=self.label_font_size, ncol=len(self.plots))

    @staticmethod
    def get_instance_title(lna_ut: int, num_of_lnas: int, lna_id: int, 
                           stage_ut: int, num_of_stages: int) -> str:
        """Returns lna and stage details for plot title"""
        inst_title = f'LNA {lna_ut} of {num_of_lnas} ' + \
                     f'(LNA ID: {lna_id}) Stage {stage_ut} of ' + \
                     f'{num_of_stages}'
        return inst_title


class BiasAccuracyPlot(CartesianPlot):
    """Plot to show how accurate the measured bias is to the set."""
    _fig = _get_std_fig()
    _axis_count = 0
    _axis_pos = []
    _fig_grid = None

    def __init__(self, set_bias: data_structs.Biases, 
                 meas_bias: data_structs.Biases, 
                 session_settings: data_structs.SessionSettings, 
                 plot_inst: PlotInstDetails) -> None:

        # region Setup figure.
        if BiasAccuracyPlot._axis_count == 0:
            
            BiasAccuracyPlot._fig_grid = gridspec.GridSpec(
                ncols=session_settings.num_of_stages, 
                nrows=session_settings.num_of_lnas, 
                figure=BiasAccuracyPlot._fig)
            
            BiasAccuracyPlot._axis_pos = list(
                itertools.product(range(session_settings.num_of_lnas),
                                  range(session_settings.num_of_stages)))

        self.ax = BiasAccuracyPlot._fig.add_subplot(
            BiasAccuracyPlot._fig_grid[
                BiasAccuracyPlot._axis_pos[
                    BiasAccuracyPlot._axis_count]])
        # endregion

        # region Set up super.
        if plot_inst.lna_ut == 1:
            lna_id = session_settings.lna_1_id
        elif plot_inst.lna_ut == 2:
            lna_id == session_settings.lna_2_id

        inst_title = super().get_instance_title(
            plot_inst.lna_ut, session_settings.num_of_lnas, 
            lna_id, plot_inst.stage_ut, 
            session_settings.num_of_stages)

        self.title = f'Bias Accuracy Plot\n{inst_title})}'

        super().__init__(self.ax, self.title)

        xminmax = [min([*meas_bias.dvs, *set_bias.dvs]) - 0.1, 
                   max([*meas_bias.dvs, *set_bias.dvs]) + 0.1]
        yminmax = [min([*meas_bias.dis, *set_bias.dis]) - 1, 
                   max([*meas_bias.dis, *set_bias.dis]) + 1]

        super().set_axis(XYAxisVars(
            'Drain Voltage (V)', 'Drain Current (mA)',
            AxisMinMax(xminmax[0], xminmax[1]), 
            AxisMinMax(yminmax[0], yminmax[1])))
        # endregion

        # region Plot points on scatter graph.
        for di in meas_bias.dis:
            for dv in meas_bias.dvs:
                self.axis.scatter(dv, di, color='red', marker='x', 
                                  linewidths=self.pixel_line_width)
        for di in set_bias.dis:
            for dv in set_bias.dvs:
                self.axis.scatter(dv, di, color='blue', marker='o', 
                                  linewidths=self.pixel_line_width)
        # region Plot points on scatter graph.

        # region Add legend.
        super().add_legend()
        # endregion

        BiasAccuracyPlot._axis_count += 1

    @staticmethod
    def  show_bias_acc_plot():
        plt.show()


class NoiseBiasPlot(CartesianPlot):
    """Noise over drain drain current."""
    _fig = _get_std_fig()
    _axis_di_count = 0
    _axis_dv_count = 0
    _axis_dv_pos = []
    _axis_di_pos = []
    _fig_grid = None
    
    def __init__(self, stage_avg_noises: np.array, 
                 stage_bias_plot_biases: data_structs.BiasPlotBiases, 
                 session_settings: data_structs.SessionSettings, 
                 v_or_i: str, plot_inst: PlotInstDetails) -> None:


        # region Setup figure.
        if NoiseBiasPlot._axis_di_count == 0 and \
                NoiseBiasPlot._axis_dv_count == 0:
            
            NoiseBiasPlot._fig_grid = gridspec.GridSpec(
                ncols=2, nrows=session_settings.num_of_stages, 
                figure=NoiseBiasPlot._fig)

            for row in range(session_settings.num_of_stages):
                NoiseBiasPlot._axis_dv_pos.append([row, 0])
                NoiseBiasPlot._axis_di_pos.append([row, 1])
            
        if v_or_i == 'v':
            self.ax = NoiseBiasPlot._fig.add_subplot(
                NoiseBiasPlot._fig_grid[
                    NoiseBiasPlot._axis_dv_pos[
                        NoiseBiasPlot._axis_dv_count]])
            NoiseBiasPlot._axis_dv_count += 1

        elif v_or_i == 'i':
            self.ax = NoiseBiasPlot._fig.add_subplot(
                NoiseBiasPlot._fig_grid[
                    NoiseBiasPlot._axis_di_pos[
                        NoiseBiasPlot._axis_di_count]])
            NoiseBiasPlot._axis_di_count += 1
        # endregion


        # region Get subplot title.
        inst_title = super().get_instance_title(
            plot_inst.lna_ut, session_settings.num_of_lnas, 
            session_settings.lna_id, plot_inst.stage_ut, 
            session_settings.num_of_stages)

        if v_or_i == 'v':
            self.title = f'Average Noise Temperature (K) over Drain Voltages\n{inst_title}'
        elif v_or_i == 'i':
            self.title = 'Average Noise Temperature (K) Over Drain Currents\n{inst_title}'
        # endregion
        
        # region Set input data into structure.



        # region Setup super.
        super().__init__(self.ax, self.title)
        
        for v_or_i == 'v':


        noise_avg_min = 
        noise_avg_max = 

        # region Configure subplot.
        if v_or_i == 'v':
            super().set_axis(XYAxisVars(
                'Drain Voltage (V)', 'Average Noise Temperature (K)',
                AxisMinMax(), AxisMinMax()))

        elif v_or_i == 'i':
            super().set_axis(XYAxisVars(
                'Drain Current (mA)', 'Average Noise Temperature (K)',
                AxisMinMax(), AxisMinMax()))
        
        super().add_legend()
        # endregion

class GainBiasPlot(CartesianPlot):
    """Gain over drain voltage, and drain current, 2 figures."""
    title = 'Average Gain (dB) Over Drain Voltages & Currents'
    def __init__(self, avg_gains, biases):
        super().__init__(self.title)
        super().set_axis(XYAxisVars())
        super().add_dataset(CartesianDataset())
        super().add_legend()


class GainOverSweep(CartesianPlot):
    """Gain over sweep position."""
    title = 'Average Gain (dB) Over Sweep Position'
    def __init__(self, avg_gains, number_of_points):
        super().__init__(self.title)
        super().set_axis(XYAxisVars())
        super().add_dataset(CartesianDataset())
        super().add_legend()


class NoiseOverSweep(CartesianPlot):
    """Noise over sweep position."""
    title = 'Average Noise Temperature (K) Over Sweep Position'
    def __init__(self, avg_noises, number_of_points):
        super().__init__(self.title)
        super().set_axis(XYAxisVars())
        super().add_dataset(CartesianDataset())
        super().add_legend()


class LoadTempsOverFreq(CartesianPlot):
    """All load over loops for hot and cold loops."""
    title = 'Load Temperature Over Frequency For All Positions'
    def __init__(self, load_temps, freqs):
        super().__init__(self.title)
        super().set_axis(XYAxisVars())
        position_datasets = []
        for position in position_datasets:
            super().add_dataset(CartesianDataset())      
        super().add_legend()


class AllNoiseFreq(CartesianPlot):
    title = ''
    def __init__(self):
        super().__init__(self.title)
        position_datasets = []
        for position in position_datasets:
            super().add_dataset()
        super().set_axis_limits()
        super().add_legend()


class AllGainFreq(CartesianPlot):
    title = ''
    def __init__(self):
        super().__init__(self.title)
        position_datasets = []
        for position in position_datasets:
            super().add_dataset()
        super().set_axis_limits()
        super().add_legend()
# endregion


class MapData:
    """Class to handle data which is presented in a heatmap.
    
    Attributes:
        x_data (np.array): X axis data.
        y_data (np.array): Y axis data.
        plot_data (np.array): Data to present on heatmap.
        data_min (float): Map colour is grey under this value.
    """
    def __init__(self, x_data, y_data, plot_data, data_min):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_data = plot_data
        self.data_min = data_min

    def plot_color_map_data(self, plot_vars: PlotVars, 
                            lna_id: int, stage_number: int) -> None:
        """Plots a heat map of the data stored in an instance."""

        # region Create a standard format figure.
        fig, axis = standard_plot_setup(
            f'LNA {lna_id} Stage {stage_number}\nMap of Average Gain (dB) over Bias Positions\n',
            'Drain Current / mA', 'Drain Voltage / V', plot_vars)
        # endregion

        # region Format figure for heatmap specifically.
        fig, axis = self.format_heatmap(fig, axis, plot_vars)
        # endregion

        # region Prepare colours for colour map, set under min to grey.
        set_cmap = plt.get_cmap('inferno')
        set_cmap.set_under('grey')
        # endregion

        # region Create and format heatmap to go on figure.
        im = axis.imshow(
            self.reshaped_data, cmap=set_cmap,
            extent=(self.x_data.min(), self.x_data.max(),
                    self.y_data.min(), self.y_data.max()),
            aspect='auto',
            interpolation='spline36')
        
        for spine in im.axes.spines.values():
            spine.set_edgecolor(plot_vars.font_colour)
        # endregion

        # region Configure the colourbar
        self._set_color_bar(fig, axis, 'Gain (dB)', plot_vars, set_cmap)
        # endregion
        
        plt.show()

    def _format_heatmap(
            self, fig: plt.Figure, axis: plt.Axes, 
            plot_vars: PlotVars) -> tuple[plt.Figure, plt.Axes]:
        """Prepares a figure for a heatmap."""
        # region Set x and y gridlines.
        axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.grid(linestyle='--', which='major', 
                  linewidth=plot_vars.pixel_line_width - 0.4)
        axis.grid(linestyle='--', which='minor', 
                  linewidth=plot_vars.pixel_line_width - 0.4)
        # endregion

        return fig, axis


    def _set_color_bar(
            self, fig: plt.Figure, axis: plt.Axes, label: str, 
            plot_vars: PlotVars, set_cmap: cmap.Colormap) -> None:
        """Creates and configures a colourbar for a heatmap figure."""
        # region Create colorbar 
        cb = fig.colorbar(
            cmap.ScalarMappable(
                norm=colors.Normalize(
                    vmin=12.5,
                    vmax=self.plot_data.max()),
                cmap=set_cmap),
                ax=axis)
        # endregion

        # region Set labels, and format for dark background.
        cb.set_label(label, color=plot_vars.font_colour)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), 
                 color=plot_vars.font_colour)
        cb.outline.set_edgecolor(plot_vars.font_colour)
        cb.ax.yaxis.set_tick_params(color=plot_vars.font_colour)
        # endregion

    def organise_set_data(self):
        """Prepares data for map over set (not meas) bias positions.
        """

        num_x = len(np.unique(self.x_data))
        num_y = len(np.unique(self.y_data))
        
        # region Scan through data placing data in correct row/column.
        # Correct formatting for a 3x3 sweep is using data indexes:
        #       Lo DI   Hi DI
        # Lo DV   0,  1,  2
        #         3,  4,  5
        # Hi DV   6,  7,  9
        reshaped_data = []
        i = 0
        while i < num_y:
            j=0
            row = []
            while j < num_x:
                row.append(
                    self.plot_data[
                        len(self.plot_data)-((i*num_x) + (num_x-j))])
                j+=1
            reshaped_data.append(row)
            i+=1
        self.reshaped_data = np.array(reshaped_data)
        # endregion

    @property
    def reshaped_data(self) -> np.array:
        """The input data shaped for input onto imshow (colormap)."""
        return self._reshaped_data
        
    @reshaped_data.setter
    def reshaped_data(self, value: np.array) -> None:
        self._reshaped_data = value


class GainMapData(MapData):
    """A subclass of map data for the gain heatmaps."""
    def __init__(self, drain_currents: np.array, drain_voltages: np.array, 
                 avg_gains: np.array, is_set: bool = False) -> None:
        """Constructor for GainMapData class.
        
        Args:
            drain_currents: Drain currents in sweep.
            drain_voltages: Drain voltages in sweep.
            avg_gains: Average gains for each bias position.
            is_set: Whether the bias data is set or measured.
        """
        super().__init__(drain_currents, drain_voltages, avg_gains, 12.5)
        if is_set:
            super().organise_set_data()


class NoiseMapData(MapData):
    """A subclass of map data for the noise heatmaps."""
    def __init__(self, drain_currents: np.array, drain_voltages:np.array,
                 avg_noises: np.array, is_set: bool = False) -> None:
        """Constructor for NoiseMapData class.
        
        Args:
            drain_currents: Drain currents in sweep.
            drain_voltages: Drain voltages in sweep.
            avg_noises: Average noises for each bias position.
            is_set: Whether the bias data is set or measured.
        """
        super().__init__(drain_currents, drain_voltages, avg_noises, 15)
        if is_set:
            super().organise_set_data()
    
