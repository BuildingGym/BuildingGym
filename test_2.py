from epluspy import idf_editor
from epluspy import idf_simu
import os
import json
from random import random

class ep_simu(idf_simu.IDF_simu):
    def control_fun(self, senstor_t):
            return [24, 0]

if __name__ == '__main__':
    idf_file = 'Main-PV-v4_ForTrain.idf'
    epw_file = 'SGP_Singapore.486980_IWEC.epw'
    output_path = 'test\\'
    epjson = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
    with open(epjson, 'r') as f:
        data = json.load(f)
    myidf = idf_editor.IDF(idf_file, epw_file, output_path)
    myidf.edit('AirLoopHVAC', 'BCA', Design_Supply_Air_Flow_Rate = 50)
    myidf.add('People', class_name='BCA', field_data = ['BCA-2','Block3:Zone2', 'Office_OpenOff_Occ', '','','','','','','On','','','','','','','','','','','','','',''])
    myidf.delete_class('airloophvac', class_name= 'BCA')
    myidf.run_period('2018-07-03', '2018-08-05')
    # myidf.write_idf_file()
    myidf = ep_simu(idf_file, epw_file, output_path, '2018-07-03', '2018-09-05', 2, True, True)
    myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'BCA',
                      Other_Equipment_Total_Heating_Rate = ['BLOCK1:ZONE1 EQUIPMENT GAIN 1', 'BLOCK2:ZONE1 EQUIPMENT GAIN 1'],
                      Zone_Mean_Air_Temperature = 'BLOCK3:ZONE1')
    myidf.actuator_call(Schedule_Value = [['ALWAYS 24', 'Schedule:Compact']], Venting_Opening_Factor = [['BLOCK3:ZONE1_WALL_4_0_0_1_0_2_WIN', 'AirFlow Network Window/Door Opening']])
    myidf.run()
    myidf.save()
    # myidf.save('C:\\Users\\Xilei Dai\\Documents\\MATLAB\\sdf.xlsx')
    # myidf.write_idf_file()
    a = 1