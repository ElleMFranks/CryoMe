# region Import modules.
from __future__ import annotations
from dataclasses import dataclass
import pathlib
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.colors as colors
from matplotlib import ticker
import pandas as pd

import replot
# endregion

class MapData:
    """Class to handle data which is presented in a heatmap.
    
    Attributes:
        x_data (np.array): X axis data.
        y_data (np.array): Y axis data.
        plot_data (np.array): Data to present on heatmap.

    """
    def __init__(self, x_data, y_data, plot_data):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_data = plot_data

    def plot_color_map_data(self, plot_vars: replot.PlotVars, 
                            lna_id: int, stage_number: int) -> None:
        
        x_label = 'Drain Current / mA'
        y_label = 'Drain Voltage / V'
        #title = f'LNA {lna_id} Stage {stage_number}\nMap of Average Gain (dB) over Bias Positions\n'
        title = f'LNA {lna_id} Stage {stage_number}\nMap of Average Noise Temperature (K) over Bias Positions\n'
        fig, axis = plt.subplots()

        fig.set_facecolor(plot_vars.colours.dark)

        axis.set_title(title,
                       color=plot_vars.font_colour,
                       fontsize=plot_vars.label_font_size+1)

        axis.set_xlabel(x_label, 
                        color=plot_vars.font_colour, 
                        fontsize=plot_vars.label_font_size)

        axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        axis.grid(linestyle='--', which='major', linewidth=plot_vars.pixel_line_width - 0.4)
        axis.grid(linestyle='--', which='minor', linewidth=plot_vars.pixel_line_width - 0.4)

        axis.set_ylabel(y_label,
                        color=plot_vars.font_colour,
                        fontsize=plot_vars.label_font_size)

        set_cmap = plt.get_cmap('inferno')
        set_cmap.set_under('grey')

        im = axis.imshow(
            self.reshaped_data, cmap=set_cmap,
            extent=(self.x_data.min(), self.x_data.max(),
                    self.y_data.min(), self.y_data.max()),
            aspect='auto',
            interpolation='spline36'
            #interpolation='gaussian'
            )
        
        for spine in im.axes.spines.values():
            spine.set_edgecolor(plot_vars.font_colour)

        axis.tick_params('both', 
                         colors=plot_vars.font_colour, 
                         labelsize=plot_vars.label_font_size)
        
        anom_trim = []
        for datum in self.plot_data:
            if datum > 5:
                anom_trim.append(datum)

        anom_trim = np.array(anom_trim)

        if len(anom_trim) == len(self.plot_data):
            v_center = self.plot_data.mean()
        else:
            v_center = anom_trim.min()

        # cb = fig.colorbar(
        #     cmap.ScalarMappable(
        #         norm=colors.Normalize(
        #             vmin=12.5,
        #             vmax=self.plot_data.max()),
        #         cmap=set_cmap),
        #         ax=axis)
        
        cb = fig.colorbar(
            cmap.ScalarMappable(
                norm=colors.Normalize(
                    vmin=self.plot_data.min(),
                    #vmin=30,
                    vmax=self.plot_data.max()),
                cmap=set_cmap),
                ax=axis)
        
        cb.set_label('Gain (dB)', color=plot_vars.font_colour)
        #cb.set_label('Noise Temperature (K)', color=plot_vars.font_colour)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=plot_vars.font_colour)
        cb.outline.set_edgecolor(plot_vars.font_colour)
        cb.ax.yaxis.set_tick_params(color=plot_vars.font_colour)
        
        plt.tight_layout()
        plt.show()

    def organise_set_data(self):
        num_x = len(np.unique(self.x_data))
        num_y = len(np.unique(self.y_data))
        
        reshaped_data = []
        i = 0
        while i < num_y:
            j=0
            row = []
            while j < num_x:
                row.append(self.plot_data[len(self.plot_data)-((i*num_x) + (num_x-j))])
                j+=1
            reshaped_data.append(row)
            i+=1
        self.reshaped_data = np.array(reshaped_data)

    @property
    def reshaped_data(self) -> np.array:
        return self._reshaped_data
        
    @reshaped_data.setter
    def reshaped_data(self, value: np.array) -> None:
        self._reshaped_data = value

class GainMapData(MapData):
    def __init__(self, drain_currents, drain_voltages, avg_gains, is_set: bool = False):
        super().__init__(drain_currents, drain_voltages, avg_gains)
        if is_set:
            super().organise_set_data()

class NoiseMapData(MapData):
    def __init__(self, drain_currents, drain_voltages, avg_noises):
        super().__init__(drain_currents, drain_voltages, avg_noises)
    
def main():
    input_session = 130
    lna_id = 36
    #input_res_log_path = pathlib.Path('C:\\Users\\m40046ef\\Documents\\Software Development\\CryoMe\\results\\Notable Sessions\\Caruso Results Log.csv')
    #input_set_log_path = pathlib.Path('C:\\Users\\m40046ef\\Documents\\Software Development\\CryoMe\\results\\Notable Sessions\\Caruso Settings Log.csv')

    #input_res_log_path = pathlib.Path('C:\\Users\\Lab\\Documents\\Caruso\\Caruso Results Log.csv')
    #input_set_log_path = pathlib.Path('C:\\Users\\Lab\\Documents\\Caruso\\Caruso Settings Log.csv')

    input_res_log_path = pathlib.Path('C:\\Users\\Lab\\Documents\\CryoMe Github\\results\\Caruso\\Caruso Results Log.csv')
    input_set_log_path = pathlib.Path('C:\\Users\\Lab\\Documents\\CryoMe Github\\results\\Caruso\\Caruso Settings Log.csv')
 
    res_log_data = np.array(pd.read_csv(input_res_log_path, header=2, error_bad_lines=False))
    set_log_data = np.array(pd.read_csv(input_set_log_path, header=1, error_bad_lines=False))

    res_session_ids = res_log_data[:,2]
    set_session_ids = set_log_data[:,2]
    input_res_log_data = []
    input_set_log_data = []

    colours = replot.PlotColours(
        '#181818', '#eeeeee', '#030bfc', '#057510', '#f74600')
    font_colour = colours.light
    pixel_line_width = 0.7
    label_font_size = 12
    plot_vars = replot.PlotVars(
        label_font_size, pixel_line_width, font_colour, colours)
    

    for i, session_id in enumerate(res_session_ids):
        if not math.isnan(session_id):
            if int(session_id) == input_session:
                input_res_log_data.append(res_log_data[i,:])

    for i, session_id in enumerate(set_session_ids):
        if not math.isnan(session_id):
            if int(session_id) == input_session:
                input_set_log_data.append(set_log_data[i,:])

    input_res_log_data = np.array(input_res_log_data)
    input_set_log_data = np.array(input_set_log_data)

    stage_1_end_index = int(len(input_res_log_data)/2)
    #stage_1_end_index = int(len(input_res_log_data))
    
    #For Ave Gain
    #stage_1_avg_gains = input_res_log_data[:stage_1_end_index,16]
    #stage_2_avg_gains = input_res_log_data[stage_1_end_index:,16]
    
    #For Ave Noise
    stage_1_avg_gains = input_res_log_data[:stage_1_end_index,21]
    stage_2_avg_gains = input_res_log_data[stage_1_end_index:,21]
    
    #For Ave Cold Power
    #stage_1_avg_gains = input_res_log_data[:stage_1_end_index,2]
    #stage_2_avg_gains = input_res_log_data[stage_1_end_index:,2]

    #For Bias Maps
    #stage_1_avg_gains = input_set_log_data[:stage_1_end_index,37]
    #stage_2_avg_gains = input_set_log_data[stage_1_end_index:,40]

    stage_1_meas_dvs = input_set_log_data[:stage_1_end_index,43]
    stage_1_meas_dis = input_set_log_data[:stage_1_end_index,44]
    stage_2_meas_dvs = input_set_log_data[stage_1_end_index:,46]
    stage_2_meas_dis = input_set_log_data[stage_1_end_index:,47]

    stage_1_set_dvs = input_set_log_data[:stage_1_end_index,23]
    stage_1_set_dis = input_set_log_data[:stage_1_end_index,24]
    stage_2_set_dvs = input_set_log_data[stage_1_end_index:,26]
    stage_2_set_dis = input_set_log_data[stage_1_end_index:,27]  

    #stage_1_set_dvs = input_set_log_data[:stage_1_end_index,25]
    #stage_1_set_dis = input_set_log_data[:stage_1_end_index,26] 
    
    limit = 60
    limitlow = 0
    for i in range(len(stage_1_avg_gains)):
        if stage_1_avg_gains[i] > limit:
            stage_1_avg_gains[i] = limit
        if stage_1_avg_gains[i] < limitlow:
            stage_1_avg_gains[i] = limitlow
    for i in range(len(stage_2_avg_gains)):
        if stage_2_avg_gains[i] > limit:
            stage_2_avg_gains[i] = limit
        if stage_2_avg_gains[i] < limitlow:
            stage_2_avg_gains[i] = limitlow
            
    # stage_1_meas_heatmap = GainMapData(
    #     stage_1_meas_dis, stage_1_meas_dvs, stage_1_avg_gains)

    # stage_2_meas_heatmap = GainMapData(
    #     stage_2_meas_dis, stage_2_meas_dvs, stage_2_avg_gains)

    stage_1_set_heatmap = GainMapData(
        stage_1_set_dis, stage_1_set_dvs, stage_1_avg_gains, True)
    
    stage_2_set_heatmap = GainMapData(
        stage_2_set_dis, stage_2_set_dvs, stage_2_avg_gains, True)

    #For Stage 2 only sweeps
    #stage_2_set_heatmap = GainMapData(
    #    stage_2_set_dis, stage_2_set_dvs, stage_1_avg_gains, True)
    
    # stage_1_set_heatmap = GainMapData(
    #      stage_1_set_dvs, stage_1_set_dis, stage_1_avg_gains, True)
    
    # stage_2_set_heatmap = GainMapData(
    #     stage_2_set_dvs, stage_2_set_dis, stage_2_avg_gains, True)
    
    stage_1_set_heatmap.plot_color_map_data(plot_vars, lna_id, 1)
    stage_2_set_heatmap.plot_color_map_data(plot_vars, lna_id, 2)
    
    #print(stage_1_set_heatmap.reshaped_data)
    
    # plt.figure()
    # plt.plot(stage_1_set_dvs)
    # plt.plot(stage_1_meas_dvs)
    
    # plt.figure()
    # plt.plot(stage_1_set_dis)
    # plt.plot(stage_1_meas_dis)
    
    # plt.figure()
    # plt.plot(stage_2_set_dvs)
    # plt.plot(stage_2_meas_dvs)
    
    # plt.figure()
    # plt.plot(stage_2_set_dis)
    # plt.plot(stage_2_meas_dis)
    
    plt.figure()
    plt.plot(stage_1_avg_gains)
    plt.plot(stage_2_avg_gains)
    

    print()
    



if __name__ == "__main__":
    main()