import os
import sys
sys.path.append('C:/EnergyPlusV9-4-0')
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
from epluspy.idf_editor import IDF
from datetime import datetime
import pandas as pd
import re

class IDF_simu(IDF):
    def __init__(self, idf_file, epw_file, output_path, start_date, end_date) -> None:
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
        self.start_date = start_date
        self.end_date = end_date
        if type(start_date) == str or type(start_date) == str:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        assert type(start_date) == type(datetime.strptime('1993-01-02', '%Y-%m-%d').date()), 'Please check the format of the start date'
        assert type(end_date) == type(datetime.strptime('1995-10-23', '%Y-%m-%d').date()), 'Please check the format of the end date'
        self._dry_run()
        self._get_edd()
        self._get_rdd()
        self._get_sensor_list()

    def run(self):
        self.run_period(self.start_date, self.end_date)
        if self._update == 1 or not os.path.exists(os.path.join(self.output_path, 'output.idf')):
            print('\033[95m'+'Save the latest modle first, please wait for a while ....'+'\033[0m')
            self.write_idf_file(self.output_path)
        ep_file_path = os.path.join(self.output_path, 'EP_file')
        if not os.path.exists(ep_file_path):
            os.mkdir(ep_file_path)
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.api.runtime.run_energyplus(self.state , ['-d', ep_file_path, '-w', self.epw_file,
                                                      os.path.join(self.output_path, 'output.idf')])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)
    
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
        assert os.path.exists(rdd_file), 'rdd file does not exist, please check'
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
        assert os.path.exists(edd_file), 'edd file does not exist, please check'
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

    def sensor(self, **kwargs):
        pass

    def actuator(self, **kwargs):
        pass