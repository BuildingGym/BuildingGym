import os
import sys
sys.path.append('C:/EnergyPlusV9-4-0')
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
from epluspy.idf_editor import IDF

class IDF_simu(IDF):
    def __init__(self, idf_file, epw_file, output_path) -> None:
        super().__init__(idf_file, epw_file, output_path)
        self._dry_run()

    def run(self):
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.api.runtime.run_energyplus(self.state , ['-d', self.output_path, '-w', self.epw_file, self.idf_file])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)
    
    def _dry_run(self):
        print('\033[95m'+'Perform a short-term dry run to get rdd and idd infomation....'+'\033[0m')
        dry_run_path = os.path.join(self.output_path, '_dry run')
        if not os.path.exists(dry_run_path):
            os.mkdir(dry_run_path)        
        self.run_period('2018-01-02', '2018-01-06')
        if not 'output:variabledictionary'.upper() in self.idf_dic:
            self.add('output:variabledictionary', 'rdd', ['regular', 'Name'])
        if not 'output:EnergyManagementSystem'.upper() in self.idf_dic:
            self.add('output:EnergyManagementSystem', 'ems', ['Verbose', 'Verbose', 'Verbose'])            
        self.write_idf_file(dry_run_path)
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.api.runtime.run_energyplus(self.state , ['-d', dry_run_path, '-w', self.epw_file, os.path.join(dry_run_path, 'output.idf')])
        self.api.runtime.clear_callbacks()
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)        


    def sensor(self, **kwargs):
        pass

    def actuator(self, **kwargs):
        pass