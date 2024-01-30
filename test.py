from epluspy import idf_editor
from epluspy import idf_simu
import os
import json

if __name__ == '__main__':
    idf_file = 'Main-PV-v4_ForTrain.idf'
    epw_file = 'USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw'
    output_path = 'test\\'
    epjson = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
    with open(epjson, 'r') as f:
        data = json.load(f)

    myidf = idf_editor.IDF(idf_file, epw_file, output_path)
    myidf.edit('AirLoopHVAC', 'BCA', Design_Supply_Air_Flow_Rate = 50)
    myidf.add('People', class_name='BCA', field_data = ['BCA-2','Block3:Zone2', 'Office_OpenOff_Occ', '','','','','','','On','','','','','','','','','','','','','',''])
    myidf.delete_class('airloophvac', class_name= 'BCA')
    myidf.run_period('2018-02-03', '2018-03-05')
    # myidf.write_idf_file()
    myidf = idf_simu.IDF_simu(idf_file, epw_file, output_path, '2018-02-03', '2018-03-05', 2, True, False)
    myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'BCA', Other_Equipment_Total_Heating_Rate = ['BLOCK1:ZONE1 EQUIPMENT GAIN 1', 'BLOCK2:ZONE1 EQUIPMENT GAIN 1'])
    myidf.actuator_call(Schedule_Value = [['OFF 24/7', 'Schedule:Compact'], ['ON 24/7', 'Schedule:Compact']])
    myidf.run()
    myidf.save()
    myidf.save('C:\\Users\\xilei\\Documents\\Zoom\\sdf.xlsx')
    # myidf.write_idf_file()
    a = 1