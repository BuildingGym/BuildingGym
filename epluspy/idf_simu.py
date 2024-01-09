import os
import sys
sys.path.append('C:/EnergyPlusV9-4-0')
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
from epluspy.idf_editor import IDF
from datetime import datetime
import pandas as pd
import re
from pathlib import Path

class IDF_simu(IDF):
    def __init__(self, idf_file, epw_file, output_path, start_date, end_date, n_time_step, sensing = False) -> None:
        """
        idf_file: The idf file path for energyplus model
        epw_file: The epw weather file for simulation
        output_path: The output folder path for output results
        start_date/end_date: Datetime.date class or string with format "yyyy-mm-dd", e.g. 2018-01-01.
        """        
        super().__init__(idf_file, epw_file, output_path)
        assert os.path.exists(output_path), f'{output_path} does not exist'
        assert os.path.exists(idf_file), f'{idf_file} does not exist'
        assert os.path.exists(epw_file), f'{epw_file} does not exist'
        self.sensor_def = False
        self.start_date = start_date
        self.end_date = end_date
        self.n_time_step = n_time_step
        self.set_time_step(n_time_step)
        self.sensing = sensing
        self.sensor_index = 0
        if self.sensing:
            self.sensor_dic = {}
        if type(self.start_date) == str or type(self.start_date) == str:
            self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d').date()
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        assert type(self.start_date) == type(datetime.strptime('1993-01-02', '%Y-%m-%d').date()), 'Please check the format of the start date'
        assert type(self.end_date) == type(datetime.strptime('1995-10-23', '%Y-%m-%d').date()), 'Please check the format of the end date'
        self.n_days = (self.end_date - self.start_date).days
        self.total_step = (self.n_days + 1) * 24 * n_time_step
        self._dry_run()
        self._get_edd()
        self._get_rdd()
        self._get_sensor_list()

    def run(self):
        if self.sensing:
            assert self.sensor_def, 'Please make sure you have correcttly define the sensor using sensor()'        
        self.run_period(self.start_date, self.end_date)
        if self._update == 1 or not os.path.exists(os.path.join(self.output_path, 'output.idf')):
            print('\033[95m'+'Save the latest model first, please wait for a while ....'+'\033[0m')
            self.write_idf_file(self.output_path)
        ep_file_path = os.path.join(self.output_path, 'EP_file')
        if not os.path.exists(ep_file_path):
            os.mkdir(ep_file_path)
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        if self.sensing == True:
            self.api.runtime.callback_end_zone_timestep_before_zone_reporting(self.state, self._sensing)
        self.api.runtime.run_energyplus(self.state , ['-d', ep_file_path, '-w', self.epw_file,
                                                      os.path.join(self.output_path, 'output.idf')])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)
        self.sensor_dic = self.sensor_dic[-int(self.total_step):]
        ts = pd.date_range(self.start_date, self.end_date + pd.Timedelta(days = 1), freq = str(int(60/self.n_time_step))+'min')[1:]
        self.sensor_dic.insert(0, 'Time', ts)
        self.run_complete = 1
    
    def _dry_run(self):
        print('\033[95m'+'Perform a short-term dry run to get rdd and idd infomation....'+'\033[0m')
        self.dry_run_path = os.path.join(self.output_path, '_dry run')
        if not os.path.exists(self.dry_run_path):
            os.mkdir(self.dry_run_path)        
        self.run_period('2015-01-02', '2015-01-06')
        if not 'output:variabledictionary'.upper() in self.idf_dic:
            self.add('output:variabledictionary', 'rdd', ['regular', 'Name'])
        if not 'output:EnergyManagementSystem'.upper() in self.idf_dic:
            self.add('output:EnergyManagementSystem', 'ems', ['Verbose', 'Verbose', 'Verbose'])
        if not 'OutputControl:Files'.upper() in self.idf_dic:
            self.add('OutputControl:Files', output_csv = 'Yes')
        else:
            self.edit(class_type = 'OutputControl:Files', class_name = 0, output_csv = 'Yes')
        self.write_idf_file(self.dry_run_path)
        try:
            with open(os.path.join(self.dry_run_path, 'eplusout.csv')) as f:
                assert f.closed == True, f'Please check if {os.path.join(self.dry_run_path, "eplusout.csv")} is closed'
        except Exception:
            pass
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.api.runtime.run_energyplus(self.state , ['-d', self.dry_run_path, '-w', self.epw_file, os.path.join(self.dry_run_path, 'output.idf')])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)

    def _get_rdd(self):
        '''
        Get the information of rdd file
        '''
        rdd_file = os.path.join(self.output_path, '_dry run', 'eplusout.rdd')
        assert os.path.exists(rdd_file), '.rdd file does not exist, please check'
        file1 = open(rdd_file, 'r')
        rdd_info = file1.readlines()[2:]
        file1.close()
        level = []
        method = []
        sensor = []
        unit = []
        for i in rdd_info:
            j = re.split(';|,|\[|\]', i)
            level.append(j[0].strip())
            method.append(j[1].strip())
            sensor.append(j[2].strip())
            unit.append(j[3].strip())
        self.rdd_df = pd.DataFrame({'Level':level, 'Method':method, 'Sensor':sensor, 'Unit':unit})

    def _get_edd(self):
        """
        Get the information of edd file
        """
        edd_file = os.path.join(self.output_path, '_dry run', 'eplusout.edd')
        assert os.path.exists(edd_file), '.edd file does not exist, please check'
        file1 = open(edd_file, 'r')
        Lines = file1.readlines()
        file1.close()
        edd_info = [s for s in Lines if "EnergyManagementSystem:Actuator Available," in s]
        component_name = []
        component_type = []
        control_variable =[]
        unit = []
        for i in edd_info:
            j = re.split(',|\[ |\]', i)
            component_name.append(j[1].strip()) # e.g. VAV_1 Supply Equipment Outlet Node
            component_type.append(j[2].strip()) # e.g. System Node Setpoint
            control_variable.append(j[3].strip()) # e.g. Temperature Setpoint
            unit.append(j[5].strip())
        self.edd_df = pd.DataFrame({'Component_name':component_name, 'Component_type':component_type,
                                    'Control_variable':control_variable, 'Unit':unit})
    
    def _get_sensor_list(self):
        dry_run_results = pd.read_csv(os.path.join(self.dry_run_path, 'eplusout.csv'), nrows = 6)
        sensor_name_list = []
        sensor_type_list = []
        for i in dry_run_results.columns[1:]:
            i = i.split(':')
            if len(i) >= 2:
                sensor_name_list.append(':'.join(i[0:-1]))
                sensor_type_list.append('['.join(i[-1].split('[')[0:-1]).strip())
            else:
                continue

        self.sensor_list = pd.DataFrame({'sensor_name': sensor_name_list, 'sensor_type': sensor_type_list})
    
    def sensor_call(self, **kwargs):
        """
        sensor_key_name = sensor_value_name
        """
        self.sensor_key_list = []
        self.sensor_value_list = []
        for key, value in kwargs.items():
            key = key.replace('_', ' ')
            self._check_sensor(key, value)
            self.sensor_key_list.append(key)
            self.sensor_value_list.append(value)
        self.sensor_def = True
        
    def _sensing(self, state):
        sensor_dic_i = {}
        wp_flag = self.api.exchange.warmup_flag(state)
        if not self.api.exchange.api_data_fully_ready(state):
            return None
        if wp_flag == 0:
            for i in range(len(self.sensor_key_list)):
                key = self.sensor_key_list[i]
                value = self.sensor_value_list[i]
                if type(value) is not list:
                    value = [value]
                for value_i in value:
                    self.sensor_i = self.api.exchange.get_variable_handle(
                        state, key, value_i
                        )
                    # assert self.sensor_i != -1, "Fail to call sensor, please check"
                    self.sensor_data = self.api.exchange.get_variable_value(state, self.sensor_i)
                    sensor_dic_i[key+'@'+value_i] = [self.sensor_data]
            sensor_dic_i = pd.DataFrame(sensor_dic_i, index = [self.sensor_index])
            if self.sensor_index == 0:
                self.sensor_dic = sensor_dic_i
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, sensor_dic_i])
            self.sensor_index+=1

    def _check_sensor(self, key, value):
        # to check
        key = key.replace('_', ' ')
        val = key in self.rdd_df['Sensor'].values
        if not val:
            self.add('output:variable', variable_name = key, reporting_frequency = 'Timestep')
        j = np.where(self.sensor_list['sensor_type'] == key)[0]
        condi = []
        if type(value) == str:
            value = [value]
        for value_i in value:
            for i in j:
                if value_i == self.sensor_list['sensor_name'][i]:
                    condi .append(True)
                    break
        assert sum(condi) == len(value), 'Please make sure the sensor name is correct'

    def save(self, path = None):
        assert self.run_complete == 1, 'Please make sure the model ran successfully before saving results'
        if path == None:
            self.sensor_dic.to_excel(os.path.join(self.output_path, 'sensor_data.xlsx'))
        else:
            if path[-5:] == '.xlsx' or path[-4:] == '.xls':
                assert os.path.exists(Path(path).parent), "Path does not exists, please check"
                self.sensor_dic.to_excel(path)
            else:
                assert os.path.exists(path), "Path does not exists, please check"
                self.sensor_dic.to_excel(os.path.join(path, 'sensor_data.xlsx'))

    def actuator(self, **kwargs):
        pass            