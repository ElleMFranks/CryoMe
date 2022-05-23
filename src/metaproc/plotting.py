import itertools
from dataclasses import dataclass
from typing import Optional

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
    lna_ut: np.array
    stage_ut: np.array


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
    colorlist = ['tab:blue',
                 'tab:orange',
                 'tab:green',
                 'tab:red',
                 'tab:purple',
                 'tab:brown',
                 'tab:pink',
                 'tab:gray',
                 'tab:olive',
                 'tab:cyan']

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
            session_settings, current_chain_config.lna_ut)
        
        inst_title = super().get_instance_title(
            current_chain_config.lna_ut, session_settings.num_of_lnas, 
            lna_id, current_chain_config.stage_ut, 
            session_settings.num_of_stages)

        self.title = f'Bias Accuracy Plot\n{inst_title}'

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

        # region Plot points on scatter graphs
        self.axis.scatter(stage_data.all_biases.set_biases.dvs,
                          stage_data.all_biases.set_biases.dis,
                          color='blue', marker='o',
                          linewidths=self.pixel_line_width + 0.1,
                          label='Target Bias')

        self.axis.scatter(stage_data.trimmed_biases.cor_meas_biases.dvs,
                          stage_data.trimmed_biases.cor_meas_biases.dis,
                          color='#04cc0e', marker='P',
                          linewidths=self.pixel_line_width,
                          label='Valid Measured Bias')

        self.axis.scatter(stage_data.bad_biases.cor_meas_biases.dvs,
                          stage_data.bad_biases.cor_meas_biases.dis,
                          color='red', marker='x', 
                          linewidths=2*self.pixel_line_width,
                          label='Invalid Measured Bias')
        # endregion

        # region Get arrows from bad points to corresponding set point.
        arrow_args = []
        all_biases_cor_meas = stage_data.all_biases.set_biases
        bad_biases_cor_meas = stage_data.all_biases.cor_meas_biases
        all_indexes = stage_data.all_biases.indexing
        bad_indexes = stage_data.bad_biases.indexing
        for i in all_indexes:
            for j in bad_indexes:
                if i == j:
                    x_start = bad_biases_cor_meas.dvs[j]
                    x_end = all_biases_cor_meas.dvs[i]
                    x_delta = x_end - x_start
                    y_start = bad_biases_cor_meas.dis[j]
                    y_end = all_biases_cor_meas.dis[i]
                    y_delta = y_end - y_start
                    arrow_args.append([x_start, y_start, x_delta, y_delta])
        
        for i, _ in enumerate(arrow_args):
            self.axis.arrow(*arrow_args[i], linestyle=(5, (3,6)), 
                            linewidth=self.pixel_line_width-0.2, color='red',
                            length_includes_head=True,
                            head_width=0.07, head_length=0.01)
        # endregion

        # region Add legend.
        self.axis.legend(fontsize=self.label_font_size-2, framealpha=0.5)
        # endregion

        BiasAccuracyPlot._axis_count += 1

    @staticmethod
    def  show_bias_acc_plot():
        plt.show()

@dataclass
class SubplotInfo:
    axis_di_count: int
    axis_dv_count: int
    axis_dv_pos: list[int, int]
    axis_di_pos: list[int, int]

@dataclass
class ResultBiasPlotFigVars:
    fig: plt.Figure
    fig_grid: Optional[gridspec.GridSpec] = None
    subplot_info: SubplotInfo = SubplotInfo(0, 0, [], [])
    

class ResultBiasPlot(CartesianPlot):
    """Noise over drain drain current."""
    _noise_fig_vars = ResultBiasPlotFigVars(_get_std_fig())
    _gain_fig_vars = ResultBiasPlotFigVars(_get_std_fig())
    
    def __init__(self, stage_data: data_structs.StageData,
                 session_settings: data_structs.SessionSettings, 
                 current_chain_config: PlotInstDetails,
                 gain_or_noise: str) -> None:
        if gain_or_noise.lower() == 'gain':
            _fig_vars = ResultBiasPlot._gain_fig_vars
            result_str = 'Average Gain (dB)'

        elif gain_or_noise.lower() == 'noise':
            _fig_vars = ResultBiasPlot._noise_fig_vars
            result_str = 'Average Noise Temperature (K)'
        fig = _fig_vars.fig

        for v_or_i in ['v', 'i']:
            
            subplot_info = _fig_vars.subplot_info
            
            is_first_instance = bool(subplot_info.axis_di_count == 0 and 
                                     subplot_info.axis_dv_count == 0)

            # region Setup figure.
            if is_first_instance:
                ResultBiasPlot._noise_fig_vars.fig_grid = gridspec.GridSpec(
                    ncols=2, nrows=session_settings.num_of_stages)
                ResultBiasPlot._gain_fig_vars.fig_grid = gridspec.GridSpec(
                    ncols=2, nrows=session_settings.num_of_stages)

                for row in range(session_settings.num_of_stages):
                    noise_info = ResultBiasPlot._noise_fig_vars.subplot_info
                    gain_info = ResultBiasPlot._gain_fig_vars.subplot_info
                    noise_info.axis_dv_pos.append((row, 0))
                    noise_info.axis_di_pos.append((row, 1))
                    gain_info.axis_dv_pos.append((row, 0))
                    gain_info.axis_di_pos.append((row, 1))
            
            grid = _fig_vars.fig_grid
            fig.tight_layout()

            lna_id = data_structs.ChainData.lna_id_from_pos(
                session_settings, current_chain_config.lna_ut)

            inst_title = super().get_instance_title(
                current_chain_config.lna_ut, session_settings.num_of_lnas, 
                lna_id, current_chain_config.stage_ut, 
                session_settings.num_of_stages)

            if v_or_i == 'v':
                bias_str = 'Over Drain Currents (mA)'
                self.ax = fig.add_subplot(
                    grid[
                        subplot_info.axis_dv_pos[
                            subplot_info.axis_dv_count]])
                subplot_info.axis_dv_count += 1
                title = f'{result_str} {bias_str}\n{inst_title}'

            elif v_or_i == 'i':
                bias_str = 'Over Drain Voltages (V)'
                self.ax = fig.add_subplot(
                    grid[
                        subplot_info.axis_di_pos[
                            subplot_info.axis_di_count]])
                subplot_info.axis_di_count += 1
                title = f'{result_str} {bias_str}\n{inst_title}'
            # endregion
            
            # region Setup super.
            super().__init__(self.ax, title)
            # endregion

            # region Organise data.
            split_biases = self.organise_data(stage_data.all_biases, v_or_i)
            split_good_biases = []
            split_bad_biases = []
            for bias_set in split_biases:
                split_good_biases.append(stage_data.biases_from_indexes(
                    bias_set.set_biases.indexing, 'good'))
                split_bad_biases.append(stage_data.biases_from_indexes(
                    bias_set.set_biases.indexing, 'bad'))
            # endregion

            # region Plot data.
            for i, good_biases in enumerate(split_good_biases):
                if v_or_i == 'i':
                    good_drain_vars = good_biases.set_biases.dvs
                    plot_label = f'{good_biases.set_biases.dis[0]:+.2f}V'
                elif v_or_i == 'v':
                    good_drain_vars = good_biases.set_biases.dis
                    plot_label = f'{good_biases.set_biases.dvs[0]:+.2f}V'
                good_results = self.get_results(
                    stage_data, good_biases.set_biases.indexing, 
                    gain_or_noise)
                good_dvs = good_biases.set_biases.dvs
                self.axis.scatter(
                    good_drain_vars, good_results, marker='o', 
                    label='_nolegend_', color=self.colorlist[i],
                    linewidth=self.pixel_line_width)
                self.axis.plot(
                    good_drain_vars, good_results, label=plot_label,
                    linewidth=self.pixel_line_width, color=self.colorlist[i])
            
            for i, bad_biases in enumerate(split_bad_biases):
                if v_or_i == 'i':
                    bad_drain_vars = bad_biases.set_biases.dvs
                if v_or_i == 'v':
                    bad_drain_vars = bad_biases.set_biases.dis
                bad_results = self.get_results(
                    stage_data, bad_biases.set_biases.indexing, gain_or_noise)
                if i == 0:
                    scatter_label = 'Invalid Bias'
                else:
                    scatter_label = '_nolabel_'
                self.axis.scatter(
                    bad_drain_vars, bad_results, label=scatter_label,
                    linewidth=self.pixel_line_width, marker = 'x', color='r')

            for i, bias_set in enumerate(split_biases):
                if v_or_i == 'i':
                    all_drain_vars = bias_set.set_biases.dvs
                if v_or_i == 'v':
                    all_drain_vars = bias_set.set_biases.dis
                all_results = self.get_results(
                    stage_data, bias_set.set_biases.indexing, gain_or_noise)
                self.axis.plot(
                    all_drain_vars, all_results, linestyle='dashed', 
                    label='_no_label_', linewidth=self.pixel_line_width-0.2,
                    color=self.colorlist[i])

            self.axis.legend(fontsize=self.label_font_size-2, framealpha=0.5)
            # endregion

        
            # region Set axis limits and labels.
            noise_minmax = 0, 100
            gain_minmax = 0, 40
            di_min = min(stage_data.all_biases.set_biases.dis)
            di_max = max(stage_data.all_biases.set_biases.dis)
            dv_min = min(stage_data.all_biases.set_biases.dvs)
            dv_max = max(stage_data.all_biases.set_biases.dvs)

            if v_or_i == 'v' and gain_or_noise == 'noise':
                super().set_axis(XYAxisVars(
                    'Drain Current (mA)', 'Average Noise Temperature (K)',
                    AxisMinMax(di_min - 0.2, di_max + 0.2), 
                    AxisMinMax(*noise_minmax)))

            elif v_or_i == 'i' and gain_or_noise == 'noise': 
                super().set_axis(XYAxisVars(
                    'Drain Voltage (V)', 'Average Noise Temperature (K)',
                    AxisMinMax(dv_min-0.01, dv_max+0.01), 
                    AxisMinMax(*noise_minmax)))

            elif v_or_i == 'v' and gain_or_noise == 'gain':
                super().set_axis(XYAxisVars(
                    'Drain Current (mA)', 'Average Gain (dB)',
                    AxisMinMax(di_min - 0.2, di_max + 0.2), 
                    AxisMinMax(*gain_minmax)))

            elif v_or_i == 'i' and gain_or_noise == 'gain': 
                super().set_axis(XYAxisVars(
                    'Drain Voltage (V)', 'Average Gain (dB)',
                    AxisMinMax(dv_min-0.01, dv_max+0.01), 
                    AxisMinMax(*gain_minmax)))
            # endregion
            
    def show(self):
        self.plt.show()

    def get_results(self, stage_data: data_structs.StageData, 
                    bias_indexes: list[int], 
                    gain_or_noise: str) -> list[float]:
        
        result = []
        for i in bias_indexes:
            for j in stage_data.results.indexing:
                if i == j:
                    if gain_or_noise == 'noise':
                        result.append(stage_data.results.noise_temp[j])
                    elif gain_or_noise == 'gain':
                        result.append(stage_data.results.gain[j])

        return result

    def organise_data(self, bias_set: data_structs.BiasSet, 
                      v_or_i: str) -> list[data_structs.BiasSet]:
        cor_meas_biases = bias_set.cor_meas_biases
        meas_biases = bias_set.meas_biases
        set_biases = bias_set.set_biases
        cor_meas_biases.indexing = bias_set.indexing
        meas_biases.indexing = bias_set.indexing
        set_biases.indexing = bias_set.indexing

        # region Breakup and organise set biases.
        organised_set_biases = self.breakup_sort_biases(set_biases, v_or_i)
        # endregion

        # region Using index of org set biases, sort meas/cor_meas.
        organised_cor_meas_biases = []
        organised_meas_biases = []
        indexes_collection = []
        for bias_list in organised_set_biases:
            indexes = [bias.index for bias in bias_list]
            indexes_collection.append(indexes)
            sorted_cor_meas_list = []
            sorted_meas_list = []
            # loop through sublist
            for i in indexes:
                if i == cor_meas_biases.indexing[i]:
                    sorted_cor_meas_list.append(
                        cor_meas_biases.bias_from_index(i))

                if i == meas_biases.indexing[i]:
                    sorted_meas_list.append(
                        meas_biases.bias_from_index(i))

            organised_cor_meas_biases.append(sorted_cor_meas_list)
            organised_meas_biases.append(sorted_meas_list)
        
        organised_bias_sets = []
        for i, _ in enumerate(organised_set_biases):
            org_set = data_structs.Biases.from_bias_set(
                organised_set_biases[i])
            org_meas = data_structs.Biases.from_bias_set(
                organised_meas_biases[i])
            org_cor_meas = data_structs.Biases.from_bias_set(
                organised_cor_meas_biases[i])
            organised_bias_sets.append(
                data_structs.BiasSet(org_set, org_cor_meas, org_meas,
                                     indexes_collection[i]))

        return organised_bias_sets
        
        # endregion

    def breakup_sort_biases(self, biases: data_structs.Biases, 
                            v_or_i: str) -> list[list[data_structs.Bias]]:
        """Returns a sorted list of biases separated by drain v or i.
        """

        # region Get a full list of the variable to split from biases.
        sort_var = []
        if v_or_i == 'v':
            for bias in biases.bias_set:
                sort_var.append(bias.dv)
        elif v_or_i == 'i':
            for bias in biases.bias_set:
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
            for bias in biases.bias_set:
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

@dataclass
class MinMaxAvg:
    mini: list[float]
    avg: list[float]
    maxi: list[float]

class ResultOverSweep(CartesianPlot):
    """Gain over sweep position."""
    _axis_count = 0
    _fig = _get_std_fig()
    title = 'Average Gain (dB) Over Sweep Position'
    def __init__(self, chain_data: data_structs.ChainData,
                 session_settings: data_structs.SessionSettings) -> None:

        for gain_or_noise in ['gain', 'noise']:
            # region Setup figure.
            gridspec.GridSpec()
            self.ax = ResultOverSweep._fig.add_subplot()

            super().__init__(self.ax, self.title)
            # endregion

            # region Get plot data.
            mini = []
            maxi = []
            avg = []
        
            for stage in chain_data.lna_1_stages.stages_data:
                if gain_or_noise.lower() == 'gain':
                    results = stage.results.gain
                elif gain_or_noise.lower() == 'noise':
                    results = stage.results.noise
                
                mini.extend((results.minis))
                maxi.extend((results.maxis))
                avg.extend((results.avgs))
            
            
            
            
        
            if gain_or_noise == 'gain':
                y_str = 'Gain (dB)'
            elif gain_or_noise == 'noise':
                y_str = 'Noise Temperature (K)'
        
            super().set_axis(XYAxisVars(
                'Bias Point Number', f'{y_str}', 
                AxisMinMax(0, len(mini)), AxisMinMax(0, max(maxi)+10)))

            self.ax.plot(range(len(mini)), mini, label='Min')
            self.ax.plot(range(len(maxi)), maxi, label='Max')
            self.ax.plot(range(len(avg)), avg, label='Avg')
            self.ax.legend()

            plt.show()
            print('')
            # endregion

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
            plot_vars: PlotVars, set_cmap: colors.Colormap) -> None:
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
    

