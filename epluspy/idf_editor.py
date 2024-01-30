import os
import sys
sys.path.append('C:/EnergyPlusV9-4-0')
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
import json
from datetime import datetime


class IDF():
    def __init__(self, idf_file, epw_file, output_path) -> None:
        """
        For parametric simulation.
        idf_file: The idf file path for energyplus model
        epw_file: The epw weather file for simulation
        output_path: The output folder path for output results
        """
        assert os.path.exists(output_path), f'{output_path} does not exist'
        assert os.path.exists(idf_file), f'{idf_file} does not exist'
        assert os.path.exists(epw_file), f'{epw_file} does not exist'
        self.idf_file = idf_file
        self.epw_file = epw_file
        self.output_path = output_path
        self.idd = self._read_idd()        
        self.idf_dic = self._create_dic()
        self._update = 0

    # def run(self):
    #     self.api = EnergyPlusAPI()
    #     self.state = self.api.state_manager.new_state()
    #     self.api.runtime.run_energyplus(self.state , ['-d', self.output_path, '-w', self.epw_file, self.idf_file])
    #     self.api.runtime.clear_callbacks()
    #     self.api.state_manager.reset_state(self.state)
    #     self.api.state_manager.delete_state(self.state)        

    def _read_idf(self, com_mark = '!'):
        # remove comment in the idf
        with open(self.idf_file, 'r') as f:
            self.ori_idf = f.read()
            f.close()
        idf_lines = self.ori_idf.splitlines()
        new_idf = []
        idf_comment = []
        for i in idf_lines:
            if com_mark in i:
                new_line = i.split(com_mark)[0]
                new_comment = i.split(com_mark)[1]
                new_idf.append(new_line)
                idf_comment.append(new_comment)
            else:
                new_line = i.split(com_mark)[0]
                if new_line == '':
                    continue
                new_idf.append(new_line)
                idf_comment.append('')                
        return '\n'.join(new_idf)

    def _create_dic(self):
        dic_idf = {}
        self.idf_nocomment = self._read_idf()
        idf_objects = self.idf_nocomment.split(';')
        for object_i in idf_objects:
            object_i = object_i.strip()
            object_i = object_i.replace('\n','')
            item_i = object_i.split(',')
            class_type = item_i[0].strip().upper()
            if class_type == '':
                continue
            field_data = item_i[1:]
            class_idd = self.idd['properties'][class_type.upper()]
            _, _, _, _, field_datatype = self._get_idd_info(class_idd)
            # if len(field_data) < len(field_datatype):
            #     for i in range(len(field_datatype) - len(field_data)):
            #         field_data.append('')
            for i in range(len(field_data)):
                if i >= len(field_datatype):
                    break
                field_data[i] = field_data[i].strip()
                if field_datatype[i] == 'number' and field_data[i] != '':
                    field_data[i] = float(field_data[i])
            if class_type in dic_idf.keys():
                dic_idf[class_type].append(field_data)
            else:
                # new_item = [field_data]
                dic_idf[class_type] = [field_data]

        # dic_idf.pop('')
        return dic_idf

    def _read_idd(self):
        idd_path = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
        with open(idd_path, 'r') as f:
            idd = json.load(f)               
            f.close()
        self.class_list = []
        self.class_list_upper = []
        key_name = list(idd['properties'].keys())
        for i in key_name:
            # Convert idd properties to upper case
            self.class_list.append(i)
            self.class_list_upper.append(i.upper())
            idd['properties'][i.upper()] = idd['properties'].pop(i)

        return idd

    def add(self, class_type, class_name = None, field_data = None, **kwargs):
        """
        add fielde data to class
        class_type: required
        class_name: only required if there is name field in the class
        field_data: list of all value in the class, or None
        Two ways to add the class:
                    1. If you prefer to specify all field data in the class, wirte them into a list to field_data. For null field, use empty string "" to occupy the field.
                    2. If you prefer to specify according to filed name, specify them in kwargs, use '_' to replace space in the field name, e.g. Design_Supply_Air_Flow_Rate = 50
                    Note that the required field must be specified
        """
        self._update = 1
        class_type = class_type.upper()
        kw_list = []
        value_list = []
        class_idd = self.idd['properties'][class_type]
        field_name, _, field_default, field_required, field_datatype = self._get_idd_info(class_idd)
        class_type = self.class_list[self.class_list_upper.index(class_type)]        
        if 'name' in class_idd.keys():
            assert class_name is not None, "Please provide a NAME for the object, e.g. class_name = 'myClass' "
        if field_data == None:
            for key, value in kwargs.items():
                kw_list.append(key)
                value_list.append(value)
            self._write_user_object_kw(class_type, field_name, field_default, kw_list, value_list, field_datatype)
        else:
            ck_req, miss_item = self._check_require(field_data, field_name, field_required)
            assert ck_req, f'The required data ({miss_item}) is missed in field_data'
            assert len(field_name) == len(field_data), 'Please make sure all files are specified in the list, use empty string "" to occupy if the field data desired to be empty'
            self._write_user_object_list(class_type, field_name, field_data, field_datatype)
    
    def _del(self, class_type, item, method):
        self._update = 1
        class_type = class_type.upper()
        field_data_list = self.idf_dic[class_type]
        class_idd = self.idd['properties'][class_type]
        field_name, _, _, _, _ = self._get_idd_info(class_idd)        
        if method =='by_name':
            for i in item:
                assert 'name' in field_name, 'This class does not include name in the field, please delete by index'
                field_data = np.array(field_data_list)
                assert i in field_data[:,0], 'The name is not found in the field data'
                index = int(np.where(i == field_data[:,0])[0])
                field_data_list.pop(index)
        if method == 'by_index':
            item.sort(reverse = True)
            assert item[0] <= len(field_data_list), f'Index ({item[0]}) out of length ({len(field_data_list)}) in class {class_type}'
            for index in item:
                field_data_list.pop(index)

    def delete_class(self, class_type, class_name = None, class_index = None):
        self._update = 1
        class_type = class_type.upper()
        if class_name == None and class_index == None:
            a = 1
        assert class_name == None or class_index == None, 'Please either specify class_name or class_index'
        if class_name is not None:
            if type(class_name) is not list:
                class_name = [class_name]
                self._del(class_type, class_name, 'by_name')
        if class_index is not None:
            if type(class_index) is not list:
                class_index = [class_index]
                self._del(class_type, class_index, 'by_index')

    def get_info(self, class_type, class_name, field_name, class_index = None):
        pass

    def _check_require(self, field_data, field_name, field_required):
        require_index = []
        if field_required == None:
            return True, None
        for i in field_required:
            require_index.append(field_name.index(i))
        for i in require_index:
            if field_data[i].strip() == '':
                return False, field_name[i]
        return True, None

    def _get_idd_info(self, class_idd):
        """
        class_idd: upper class type, e.g. FAN:CONSTANTVOLUME
        """
        field_name = []
        field_option = []
        field_datatype = []
        field_default = []

        if 'name' in class_idd.keys():
            field_name.append('name')
            field_option.append('')
            field_default.append('')
            field_datatype.append('string') # string or number
        # get item name to field_name
        if  '^.*\\S.*$' in class_idd['patternProperties'].keys():
            class_key = '^.*\\S.*$'
        elif '.*' in  class_idd['patternProperties'].keys():
            class_key = '.*'
        else:
            raise AttributeError('class key not found')
        class_idd_properties = class_idd['patternProperties'][class_key]['properties']
        if 'required' in class_idd['patternProperties'][class_key].keys():
            field_required = class_idd['patternProperties'][class_key]['required']
        else:
            field_required = None
        for i in range(len(class_idd_properties.keys())):
            field_name_i = list(class_idd_properties.keys())[i]
            field_name.append(field_name_i)
            if 'enum' in class_idd_properties[field_name_i].keys():
                field_option.append(class_idd_properties[field_name_i]['enum'])
            else:
                field_option.append('')
            if 'default' in class_idd_properties[field_name_i].keys():
                field_default.append(class_idd_properties[field_name_i]['default'])       
            else:
                field_default.append('')
            if 'type' in class_idd_properties[field_name_i].keys():
                field_datatype.append(class_idd_properties[field_name_i]['type'])
            else:
                field_datatype.append('string')
    
        return field_name, field_option, field_default, field_required, field_datatype

    def _write_user_object_kw(self, class_type, field_name, field_default, kw_list, value_list, field_datatype):
        field_data = field_default
        for i in range(len(kw_list)):
            index = field_name.index(kw_list[i].lower().replace(' ', '_'))
            field_data[index] = value_list[i]
        self._to_dic(class_type, field_data, field_datatype)

    def _write_user_object_list(self, class_type, field_name, field_data, field_datatype):
        self._to_dic(class_type, field_data, field_datatype)
    
    def _write_object(self, class_type, field_name, field_data, file_path, mode = 'a'):
        while len(field_name) < len(field_data):
            field_name.append('Data')
        with open(file_path, mode) as f:
            output = class_type + ',\n'
            for i in range(len(field_data)):
                if i < (len(field_data)-1):
                    output = output + '\t' + str(field_data[i]).strip() + ',' + '\t' + '\t' + '\t' + '\t' + '!- ' + field_name[i] + '\n'
                else:
                    output = output + '\t' + str(field_data[i]).strip() + ';' + '\t' + '\t' + '\t' + '\t' + '!- ' + field_name[i] + '\n'
            f.write(output)
        f.close()

    def _to_dic(self, class_type, field_data, field_datatype):
        class_type = class_type.upper()
        for i in range(len(field_datatype)):
            if field_datatype[i] == 'number' and field_data[i] != '':
                field_data[i] = float(field_data[i])
        if class_type in self.idf_dic.keys():
            self.idf_dic[class_type].append(field_data)
        else:
            if class_type in self.idf_dic:
                self.idf_dic[class_type] = field_data
            else:
                self.idf_dic[class_type] = [field_data]
        
    def write_idf_file(self, file_path = os.getcwd()):
        print('\033[95m'+'===Writing idf file for output, please wait for a while.....==='+'\033[0m')
        mode = 'w'
        file_path = os.path.join(file_path,'output.idf')
        for class_type in self.idf_dic.keys():
            index = self.class_list_upper.index(class_type.upper())
            class_type = self.class_list[index]
            class_idd = self.idd['properties'][class_type.upper()]
            field_name, _, _, _, _ = self._get_idd_info(class_idd)
            field_data = self.idf_dic[class_type.upper()]
            for i in range(len(field_data)):
                self._write_object(class_type, field_name, field_data[i], file_path, mode)
                mode = 'a'
        print('\033[95m'+'===Successfully output idf file!==='+'\033[0m')
        self._update = 0

    def edit(self, class_type, class_name, **kwargs):
        """
        class_name: set it as 'All' if edit all class in this type, otherwise specify calss_name. class_name = 'All', or class_name = 'Airloop-1'
        **kwargs: write field name and value. use "_" to replace space in the field name, e.g. Design_Supply_Air_Flow_Rate = 50
        """
        self._update = 1
        class_type = class_type.upper()
        class_idd = self.idd['properties'][class_type]
        field_name, _, field_default, field_required, field_datatype = self._get_idd_info(class_idd)

        field_data_list = self.idf_dic[class_type]

        for key, value in kwargs.items():
            # To write: check value type
            key = key.lower()
            index = field_name.index(key)
            if class_name == 'All':
                for i in range(len(field_data_list)):
                    self.idf_dic[class_type][i][index] = value
                done = True
            elif type(class_name) == type(int(8)):
                    assert class_name<= len(field_data_list), f'Index of {class_name} out of range, please check the index'
                    self.idf_dic[class_type][class_name][index] = value
                    done = True                
            else:
                for i in range(len(field_data_list)):
                    if field_data_list[i][0] == class_name:
                        self.idf_dic[class_type][i][index] = value
                        done = True
                    else:
                        continue
        assert done == True, "Fail to find the class, please specify the corrct name"

    def run_period(self, start_date, end_date):
        """
        start_date: datetime.date class or string with format "yyyy-mm-dd", e.g. 2018-01-01.
        end_date: same as start date
        """
        self._update = 1
        if type(start_date) == str or type(start_date) == str:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        assert type(start_date) == type(datetime.strptime('1993-01-02', '%Y-%m-%d').date()), 'Please check the format of the start date'
        assert type(end_date) == type(datetime.strptime('1995-10-23', '%Y-%m-%d').date()), 'Please check the format of the end date'
        self.edit('RunPeriod', 'All',
                  begin_year = start_date.year,
                  begin_month = start_date.month,
                  Begin_day_of_month = start_date.day,
                  end_year = end_date.year,
                  end_month = end_date.month,
                  end_day_of_month = end_date.day)
        
    def set_time_step(self, n):
        """
        n: number of time steps in one hour
        """
        if 'timestep'.upper() in self.idf_dic:
            assert 60 % n == 0, 'Please make sure n is evenly divisible into 60'
            self.edit('timestep', 'All', number_of_timesteps_per_hour = n)
        else:
            self.add('timestep', [n])
                
    def options(self, class_type, att_name):
        pass
