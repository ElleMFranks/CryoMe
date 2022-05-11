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

    def __init__(self, stage_data: data_structs.StageData, 
                 session_settings: data_structs.SessionSettings, 
                 current_chain_config: PlotInstDetails) -> None:

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
        lna_id = data_structs.ChainData.lna_id_from_pos(
            current_chain_config, session_settings)
        
        inst_title = super().get_instance_title(
            current_chain_config.lna_ut, session_settings.num_of_lnas, 
            lna_id, current_chain_config.stage_ut, 
            session_settings.num_of_stages)

        self.title = f'Bias Accuracy Plot\n{inst_title})}'

        super().__init__(self.ax, self.title)

        xminmax = [stage_data.get_minmax_bias('v', 'min') - 0.1, 
                   stage_data.get_minmax_bias('v', 'max') + 0.1]
        yminmax = [stage_data.get_minmax_bias('i', 'min') - 1, 
                   stage_data.get_minmax_bias('i', 'max') + 1]

        super().set_axis(XYAxisVars(
            'Drain Voltage (V)', 'Drain Current (mA)',
            AxisMinMax(xminmax[0], xminmax[1]), 
            AxisMinMax(yminmax[0], yminmax[1])))
        # endregion

        # region Plot points on scatter graph.
        for di in stage_data.all_biases.set_biases.dis:
            for dv in stage_data.all_biases.set_biases.dvs:
                self.axis.scatter(dv, di, color='blue', marker='o', 
                                  linewidths=self.pixel_line_width,
                                  label='Target Bias')

        for di in stage_data.trimmed_biases.cor_meas_biases.dis:
            for dv in stage_data.trimmed_biases.cor_meas_biases.dvs:
                self.axis.scatter(dv, di, color='green', marker='+', 
                                  linewidths=self.pixel_line_width,
                                  label='Valid Measured Bias')
        
        for di in stage_data.bad_biases.cor_meas_biases.dis:
            for dv in stage_data.bad_biases.cor_meas_biases.dvs:
                self.axis.scatter(dv, di, color='red', marker='x', 
                                  linewidths=self.pixel_line_width,
                                  label='Invalid Measured Bias')
        # endregion

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
    
    def __init__(self, stage_data: data_structs.StageData,
                 session_settings: data_structs.SessionSettings, 
                 v_or_i: str, current_chain_config: PlotInstDetails) -> None:

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
        lna_id = data_structs.ChainData.lna_id_from_pos(
            session_settings, current_chain_config.lna_ut)

        inst_title = super().get_instance_title(
            current_chain_config.lna_ut, session_settings.num_of_lnas, 
            lna_id, current_chain_config.stage_ut, 
            session_settings.num_of_stages)

        if v_or_i == 'v':
            self.title = f'Average Noise Temperature (K) over Drain Voltages\n{inst_title}'
        elif v_or_i == 'i':
            self.title = 'Average Noise Temperature (K) Over Drain Currents\n{inst_title}'
        # endregion

        # region Organise data.
        self.split_all_biases = self.breakup_sort_biases(
            stage_data.all_biases.set_biases, v_or_i)
        self.split_bad_biases = self.breakup_sort_biases(
            stage_data.bad_biases.set_biases, v_or_i)
        self.split_trimmed_biases = self.breakup_sort_biases(
            stage_data.trimmed_biases.set_biases, v_or_i)
        # endregion

        # region Set axis limits and labels.
        noise_minmax = AxisMinMax
        di_minmax = AxisMinMax
        dv_minmax = AxisMinMax
        if v_or_i == 'v':
            super().set_axis(XYAxisVars(
                'Drain Voltage (V)', 'Average Noise Temperature (K)',
                noise_minmax, dv_minmax))

        elif v_or_i == 'i':
            super().set_axis(XYAxisVars(
                'Drain Current (mA)', 'Average Noise Temperature (K)',
                noise_minmax, di_minmax))
        # endregion

        # region Plot good and bad bias points using scatter.
        # endregion

        # region Plot a line using the good points only.
        # endregion

        # region Plot a dashed thin line using the bad points too.
        # endregion

        # region Setup super.
        super().__init__(self.ax, self.title)
        # region Configure subplot.
        
        
        super().add_legend()
        # endregion


    def organise_data(self, bias_set: data_structs.BiasSet, 
                      v_or_i: str) -> data_structs.BiasSet:
        cor_meas_biases = bias_set.cor_meas_biases
        meas_biases = bias_set.meas_biases
        set_biases = bias_set.set_biases

        # region Breakup and organise set biases.
        organised_set_biases = self.breakup_sort_biases(set_biases, v_or_i)
        # endregion

        # region Using index of org set biases, sort meas/cor_meas.
        organised_cor_meas_biases = []
        organised_meas_biases = []
 
        for bias_list in organised_set_biases:
            sorted_cor_meas_list = []
            sorted_meas_list = []
            # loop through sublist
            for i, in bias_list.bias.index:
                if i == cor_meas_biases.indexing[i]:
                    sorted_cor_meas_list.append(
                        cor_meas_biases.bias_from_index(i))

                if i == meas_biases.indexing[i]:
                    sorted_meas_list.append(
                        meas_biases.bias_from_index(i))
            organised_cor_meas_biases.append(sorted_cor_meas_list)
            organised_meas_biases.append(sorted_meas_list)
        # endregion

    def breakup_sort_biases(self, biases: data_structs.Biases, 
                            v_or_i: str) -> list[list[data_structs.Bias]]:
        """Returns a sorted list of biases separated by drain v or i.
        """

        # region Get a full list of the variable to split from biases.
        sort_var = []
        if v_or_i == 'v':
            for bias in biases:
                sort_var.append(bias.dv)
        elif v_or_i == 'i':
            for bias in biases:
                sort_var.append(bias.di)
        # endregion
        
        # region Create a set of the unique values in the full list.
        sort_var_set = set(sort_var)
        # endregion

        #region Separate the biases based on the created set.
        sort_var_set_arrays = []
        for var in sort_var_set:
            var_array = []
            # region For each bias if sort var in set add bias to list.
            for bias in biases:
                if bias.dv == var and v_or_i == 'v':
                    var_array.append(bias)
                elif bias.di == var and v_or_i == 'i':
                    var_array.append(bias)
            # endregion

            # region Sort result list of bias objects by alternate var.
            if v_or_i == 'i':
                sorted_var_array = sorted(
                    var_array, key=lambda bias: bias.dv)
            elif v_or_i == 'v':
                sorted_var_array = sorted(
                    var_array, key=lambda bias: bias.di)
            # endregion

            #region Append sorted, split list into output list.
            sort_var_set_arrays.append(sorted_var_array)
            # endregion
        # endregion

        return sort_var_set_arrays



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
    

