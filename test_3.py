from epluspy import idf_editor
from epluspy import idf_simu
import os
import json
from random import random


class ep_simu(idf_simu.IDF_simu):
    def control_fun(self, senstor_t):
            return [12, 10]


if __name__ == '__main__':
    idf_file = 'DB model(modified)-v6.idf'
    epw_file = 'SGP_SINGAPORE-CHANGI-AP_486980_18.epw'
    output_path = 'test\\'
    # myidf.write_idf_file()
    for i in range(10, 11):
        myidf = ep_simu(idf_file, epw_file, output_path, '2018-01-01', '2018-1-31', 2, True, True, i)
        # myidf.delete_class('Output:Variable')
        # myidf.sensor_call(Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
        #                   Site_Outdoor_Air_Wetbulb_Temperature = ['Environment'],
        #                   Plant_Supply_Side_Inlet_Temperature = 'CHW LOOP',
        #                   Plant_Supply_Side_Outlet_Temperature = 'CHW LOOP',
        #                   Plant_Supply_Side_Inlet_Mass_Flow_Rate = 'CHW LOOP',
        #                   Chiller_Evaporator_Mass_Flow_Rate = ['CHILLER', 'CHILLER 1', 'CHILLER 2'],
        #                   Chiller_Electricity_Energy = ['CHILLER', 'CHILLER 1', 'CHILLER 2'])
        myidf.sensor_call(Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                          Chiller_Electricity_Energy = ['CHILLER', 'CHILLER 1', 'CHILLER 2'],
                          Chiller_Electricity_Rate  = ['CHILLER', 'CHILLER 1', 'CHILLER 2'],
                          Plant_Supply_Side_Inlet_Temperature = 'CHW LOOP',
                          Plant_Supply_Side_Outlet_Temperature = 'CHW LOOP',
                          Chiller_Evaporator_Outlet_Temperature = ['CHILLER', 'CHILLER 1', 'CHILLER 2'],
                          Pump_Mass_Flow_Rate = ['CHW LOOP SUPPLY SIDE PUMP', 'CHW LOOP SUPPLY SIDE PUMP 1', 'CHW LOOP SUPPLY SIDE PUMP 2'])
        myidf.edit('Output:variable', class_name = 'All', reporting_frequency = 'Timestep')
        myidf.actuator_call(Schedule_Value = [['CHW LOOP CONTROL', 'Schedule:Compact'],
                                              ['CHW CHILLER CONTROL', 'Schedule:Compact']])
        myidf.run()
        myidf.save()
    # myidf.save('C:\\Users\\Xilei Dai\\Documents\\MATLAB\\sdf.xlsx')
    # myidf.write_idf_file()
    a = 1