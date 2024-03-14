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
    def __init__(self, idf_file, epw_file, output_path, start_date, end_date, n_time_step, sensing = False, control = False, runtime_id = 0) -> None:
        """
        idf_file: The idf file path for energyplus model
        epw_file: The epw weather file for simulation
        output_path: The output folder path for output results
        start_date/end_date: Datetime.date class or string with format "yyyy-mm-dd", e.g. 2018-01-01.
        """        
        super().__init__(idf_file, epw_file, output_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        assert os.path.exists(idf_file), f'{idf_file} does not exist'
        assert os.path.exists(epw_file), f'{epw_file} does not exist'
        self.sensor_def = False
        self.actuator_def = False
        self.start_date = start_date
        self.end_date = end_date
        self.n_time_step = n_time_step
        self.set_time_step(n_time_step)
        self.sensing = sensing
        self.control = control
        self.runtime_id = runtime_id
        if type(self.start_date) == str or type(self.start_date) == str:
            self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d').date()
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        assert type(self.start_date) == type(datetime.strptime('1993-01-02', '%Y-%m-%d').date()), 'Please check the format of the start date'
        assert type(self.end_date) == type(datetime.strptime('1995-10-23', '%Y-%m-%d').date()), 'Please check the format of the end date'
        self.ts = pd.date_range(self.start_date, self.end_date + pd.Timedelta(days = 1), freq = str(int(60/self.n_time_step))+'min')[1:]
        self.n_days = (self.end_date - self.start_date).days
        self.total_step = (self.n_days + 1) * 24 * n_time_step
        self._dry_run()
        self._get_edd()
        self._get_rdd()
        self._get_sensor_list()

    def run(self, epsilon = 0):
        self.epsilon = epsilon
        if self.sensing:
            self.sensor_dic = {}
        if self.control:
            self.cmd_dic = {}                    
            self.action_dic = {}                    
        self.sensor_index = 0
        self.cmd_index = 0  
        if self.sensing:
            assert self.sensor_def, 'Please make sure you have correcttly define the sensor using sensor_call()'        
        if self.control:
            assert self.actuator_def, 'Please make sure you have correcttly define the actuator using actuator_call()'              
        self.run_period(self.start_date, self.end_date)
        if self._update == 1 or not os.path.exists(os.path.join(self.output_path, 'output.idf')):
            print('\033[95m'+'Save the latest model first, please wait for a while ....'+'\033[0m')
        # self.write_idf_file(self.output_path) # To add version id
        ep_file_path = os.path.join(self.output_path, 'EP_file')
        if not os.path.exists(ep_file_path):
            os.mkdir(ep_file_path)
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        if self.sensing == True and self.control == False:
            # self.api.runtime.callback_end_zone_timestep_before_zone_reporting(self.state, self._sensing)
            if self.runtime_id == 0:
                self.api.runtime.callback_message(self.state, self._sensing)
            if self.runtime_id == 1:
                self.api.runtime.callback_inside_system_iteration_loop(self.state, self._sensing)
            if self.runtime_id == 2:
                self.api.runtime.callback_end_zone_timestep_before_zone_reporting(self.state, self._sensing)
            if self.runtime_id == 3:
                self.api.runtime.callback_end_zone_timestep_after_zone_reporting(self.state, self._sensing)
            if self.runtime_id == 4:
                self.api.runtime.callback_end_zone_sizing(self.state, self._sensing)
            if self.runtime_id == 5:
                self.api.runtime.callback_end_system_timestep_before_hvac_reporting(self.state, self._sensing)
            if self.runtime_id == 6:
                self.api.runtime.callback_end_system_timestep_after_hvac_reporting(self.state, self._sensing)
            if self.runtime_id == 7:
                self.api.runtime.callback_end_system_sizing(self.state, self._sensing)
            if self.runtime_id == 8:
                self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(self.state, self._sensing)
            if self.runtime_id == 9:
                self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.state, self._sensing)
            if self.runtime_id == 10:
                self.api.runtime.callback_begin_system_timestep_before_predictor(self.state, self._sensing)      
            if self.runtime_id == 11:
                self.api.runtime.callback_begin_new_environment(self.state, self._sensing)      
            if self.runtime_id == 12:
                self.api.runtime.callback_after_predictor_before_hvac_managers(self.state, self._sensing)      
            if self.runtime_id == 13:
                self.api.runtime.callback_after_predictor_after_hvac_managers(self.state, self._sensing)      
            if self.runtime_id == 14:
                self.api.runtime.callback_after_new_environment_warmup_complete(self.state, self._sensing)      
            if self.runtime_id == 15:
                self.api.runtime.callback_after_component_get_input(self.state, self._sensing)                                                                                                                                                                                                                                                
        if self.sensing == False and self.control == True:
            self.api.runtime.callback_end_zone_timestep_before_zone_reporting(self.state, self._control)
        if self.sensing == True and self.control == True:
            self.api.runtime.callback_end_zone_timestep_before_zone_reporting(self.state, self._sensing_ctrl)      
        self.api.exchange.request_variable(self.state , "Chiller Electricity Energy", "CHILLER")                  
        self.api.runtime.run_energyplus(self.state , ['-d', ep_file_path, '-w', self.epw_file,
                                                      os.path.join(self.output_path, 'output.idf')])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)
        if len(self.sensor_dic) >= self.total_step:
            self.sensor_dic = self.sensor_dic[-int(self.total_step):]
            if self.control:
                self.cmd_dic = self.cmd_dic[-int(self.total_step):]
                self.action_dic = self.action_dic[-int(self.total_step):]
            if not 'Time' in self.sensor_dic:
                self.sensor_dic.insert(0, 'Time', self.ts)
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
        control_type =[]
        unit = []
        for i in edd_info:
            j = re.split(',|\[ |\]', i)
            component_name.append(j[1].strip()) # e.g. VAV_1 Supply Equipment Outlet Node
            component_type.append(j[2].strip()) # e.g. System Node Setpoint
            control_type.append(j[3].strip()) # e.g. Temperature Setpoint
            # unit.append(j[5].strip())
            unit.append(j[-1].strip())
        self.edd_df = pd.DataFrame({'Component_name':component_name, 'Component_type':component_type,
                                    'Control_type':control_type, 'Unit':unit})
        self.edd_df.to_csv(os.path.join(self.output_path, '_dry run','edd.csv'))
    
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
        self.sensor_list.to_csv(os.path.join(self.output_path, '_dry run','rdd.csv'))
    
    def sensor_call(self, **kwargs):
        """
        sensor_key_name = sensor_value_name
        """
        if not self.sensing:
            print('\033[93mWARNING: you call the sensor but not activate the sensing function.\
                  Set self.sensing as True if you want to sense during simulation\033[00m')
        self.sensor_key_list = []
        self.sensor_value_list = []
        for key, value in kwargs.items():
            key = key.replace('_', ' ')
            self._check_sensor(key, value)
            self.sensor_key_list.append(key)
            self.sensor_value_list.append(value)
        self.sensor_def = True

    # def actuator_call(self, **kwargs):
    #     """
    #     sensor_key_name = sensor_value_name
    #     """
    #     if not self.sensing:
    #         print('\033[93mWARNING: you call the sensor but not activate the sensing function.\
    #               Set self.sensing as True if you want to sense during simulation\033[00m')
    #     self.control_type_sensing_list = []
    #     self.component_name_sensing_list = []
    #     self.component_type_sensing_list = []
    #     for key, value in kwargs.items():
    #         key = key.replace('__', '/')
    #         key = key.replace('_', ' ')
    #         self._check_actuator(key, value)
    #         if len(np.array(value).shape) == 1:
    #             value = [value]
    #         for value_i in value:
    #             self.control_type_sensing_list.append(key)
    #             self.component_name_sensing_list.append(value_i[0].upper())
    #             self.component_type_sensing_list.append(value_i[1])
        
    def _sensing(self, state):
        assert self.sensing, 'Please initialize sensing as "True" in IDF_simu Class if you would like to call sensor during simulation'
        sensor_dic_i = {}
        wp_flag = self.api.exchange.warmup_flag(state)
        if not self.api.exchange.api_data_fully_ready(state):
            return None
        if not wp_flag:
            # sensor_dic_i['Year'] = self.api.exchange.year(state)
            sensor_dic_i['Month'] = self.api.exchange.month(state)
            sensor_dic_i['Day'] = self.api.exchange.day_of_month(state)
            sensor_dic_i['Hour'] = self.api.exchange.hour(state)
            sensor_dic_i['Min'] = self.api.exchange.minutes(state)
            sensor_dic_i['Day_of_Week'] = self.api.exchange.day_of_week(state)
            for i in range(len(self.sensor_key_list)):
                key = self.sensor_key_list[i]
                value = self.sensor_value_list[i]
                if type(value) is not list:
                    value = [value]
                for value_i in value:                           
                    self.sensor_i = self.api.exchange.get_variable_handle(
                        state, key, value_i
                        )
                    assert self.sensor_i != -1, "SENSOR NAME ERROE, please check sensor_name for sensor_call"
                    self.sensor_data = self.api.exchange.get_variable_value(state, self.sensor_i)
                    sensor_dic_i[key+'@'+value_i] = [self.sensor_data]
            # for i in range(len(self.component_name_sensing_list)):

            #     key = self.component_type_sensing_list[i]
            #     value = self.component_name_sensing_list[i]
            #     if type(value) is not list:
            #         value = [value]
            #     for value_i in value:                           
            #         self.sensor_i = self.api.exchange.get_actuator_handle(
            #             state, self.component_type_sensing_list[i], self.control_type_sensing_list[i], self.component_name_sensing_list[i].upper()
            #             ) # component_type, control_type, actuator_key
                          
            #         assert self.sensor_i != -1, "SENSOR NAME ERROE, please check sensor_name for sensor_call"
            #         self.sensor_data = self.api.exchange.get_actuator_value(state, self.sensor_i)
            #         sensor_dic_i[key+'@'+value_i] = [self.sensor_data]

            sensor_dic_i = pd.DataFrame(sensor_dic_i, index = [self.sensor_index])
            if self.sensor_index == 0:
                self.sensor_dic = sensor_dic_i
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, sensor_dic_i])
            self.sensor_index+=1
            self.sensor_t = sensor_dic_i

    def set_agent(self, agent, input_var):
        self.agent = agent
        self.input_var = input_var

    def actuator_ctrl(self, **kwargs):
        if self.control == False:
            return
        """
        Control type = [Component Unique Name, Component Type]
        """
        if not self.control:
            print('\033[40m' + 'WARNING: you call the actuator but not activate the control function.\
                  Set self.control as True if you want to control using actuator' + '\033[00m')
        self.control_type_ctrl_list = []
        self.component_name_ctrl_list = []
        self.component_type_ctrl_list = []
        for key, value in kwargs.items():
            key = key.replace('__', '/')
            key = key.replace('_', ' ')
            self._check_actuator(key, value)
            if len(np.array(value).shape) == 1:
                value = [value]
            for value_i in value:
                self.control_type_ctrl_list.append(key)
                self.component_name_ctrl_list.append(value_i[0].upper())
                self.component_type_ctrl_list.append(value_i[1])
        self.actuator_def = True

    def _control(self, state):
        cmd_dic_i = {}
        action_dic_i = {}
        assert self.control, 'Please initialize control as "True" in IDF_simu Class if you would like to call sensor during simulation'
        assert 'control_fun' in dir(self), "please define the control function as control_fun()"
        wp_flag = self.api.exchange.warmup_flag(state)
        if not self.api.exchange.api_data_fully_ready(state):
            return None
        if wp_flag == 0:        
            com, action = self.control_fun(self.sensor_t)
            assert type(com) == list, 'The output of the control_fun() should be list'
            assert len(com) == len(self.component_type_ctrl_list), 'The length of command and the number of actuator should be same'
            for i in range(len(self.component_type_ctrl_list)):
                self.actuator_id = self.api.exchange.get_actuator_handle(
                    state,
                    self.component_type_ctrl_list[i],
                    self.control_type_ctrl_list[i],
                    self.component_name_ctrl_list[i]
                    ) # component_type, control_type, actuator_key
                self.api.exchange.set_actuator_value(state , self.actuator_id, com[i])
                cmd_dic_i[self.component_type_ctrl_list[i]+'@'+self.control_type_ctrl_list[i]+'@'+self.component_name_ctrl_list[i]] = [com[i]]
                action_dic_i[self.component_type_ctrl_list[i]+'@'+self.control_type_ctrl_list[i]+'@'+self.component_name_ctrl_list[i]] = [action[i]]
            cmd_dic_i = pd.DataFrame(cmd_dic_i, index = [self.cmd_index])
            action_dic_i = pd.DataFrame(action_dic_i, index = [self.cmd_index])
            if self.cmd_index == 0:
                self.cmd_dic = cmd_dic_i
                self.action_dic = action_dic_i
            else:
                self.cmd_dic = pd.concat([self.cmd_dic, cmd_dic_i])
                self.action_dic = pd.concat([self.action_dic, action_dic_i])
            self.cmd_index+=1

    def _sensing_ctrl(self, state):
        self._sensing(state) # sensor_t: the simulation results at timestep t
        self._control(state)

    def _check_sensor(self, key, value):
        key = key.replace('_', ' ')
        val = key in list(self.sensor_list['sensor_type'])
        if not val:
            self.add('output:variable', variable_name = key, reporting_frequency = 'Timestep')
            print('\033[93m'+'WARNING: automaically add <<' + key + '>> into Output:Variable, please make sure sensor value name is correct'+'\033[00m')
            self._update == 1
        else:
            j = np.where(self.sensor_list['sensor_type'] == key)[0]
            condi = []
            if type(value) == str:
                value = [value]
            for value_i in value:
                for i in j:
                    if value_i == self.sensor_list['sensor_name'][i]:
                        condi.append(True)
                        break
            assert sum(condi) == len(value), f'Please make sure the sensor name is correct for key "{key}"'

    def _check_actuator(self, key, value):
        key = key.replace('__', '/')
        key = key.replace('_', ' ')
        val = key in self.edd_df['Control_type'].values
        assert val, f'{key} not found in edd file, please check'
        j = np.where(self.edd_df['Control_type'] == key)[0]
        condi = []
        for value_i in value:
            for i in j:
                if value_i[0].upper() == self.edd_df['Component_name'][i] and value_i[1] == self.edd_df['Component_type'][i]:
                    condi.append(True)
                    break
        assert sum(condi) == len(value), 'Please make sure the actuator name (control type, component name, component type) is correct'

    def save(self, path = None):
        assert self.run_complete == 1, 'Please make sure the model ran successfully before saving results'
        try:
            if path == None:
                if self.sensing:
                    self.sensor_dic.to_excel(os.path.join(self.output_path, str(self.runtime_id) + '-sensor_data.xlsx'))
                if self.control:
                    self.cmd_dic.to_excel(os.path.join(self.output_path, str(self.runtime_id) + '-cmd_data.xlsx'))
            else:
                assert os.path.exists(path), "Path does not exists, please check"
                if self.sensing:
                    self.sensor_dic.to_excel(os.path.join(path, str(self.runtime_id) + '-sensor_data.xlsx'))
                if self.control:
                    self.cmd_dic.to_excel(os.path.join(path, str(self.runtime_id) + '-cmd_data.xlsx'))
                    
        except:
            print('\033[93m'+'WARNING: ===Failed to save the result files==='+'\033[0m')