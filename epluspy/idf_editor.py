import os
import sys
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
import json
sys.path.append('C:/EnergyPlusV9-4-0')


class IDF():
    def __init__(self, idf_file, epw_file, output_path) -> None:
        self.idf_file = idf_file
        self.epw_file = epw_file
        self.output_path = output_path
        self.idd = self._read_idd()        
        self.idf_dic = self._create_dic()
        # print(self.idf_nocomment)
        pass
    
    def run(self):
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.api.runtime.run_energyplus(self.state , ['-d', self.output_path, '-w', self.epw_file, self.idf_file])
        self.api.state_manager.reset_state(self.state)
        self.api.state_manager.delete_state(self.state)

    def _read_idf(self, com_mark = '!'):
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
        # f.close()
        self.class_list = []
        self.class_list_upper = []
        key_name = list(idd['properties'].keys())
        for i in key_name:
            self.class_list.append(i)
            self.class_list_upper.append(i.upper())
            idd['properties'][i.upper()] = idd['properties'].pop(i)

        return idd

    def add(self, class_type, class_name = None, field_data = None, **kwargs):
        kw_list = []
        value_list = []
        class_idd = self.idd['properties'][class_type.upper()]
        field_name, _, field_default, field_required, field_datatype = self._get_idd_info(class_idd)
        class_type = self.class_list[self.class_list_upper.index(class_type.upper())]        
        if 'name' in class_idd.keys():
            assert class_name is not None, "Please provide a NAME for the object, e.g. class_name = 'myClass' "
        if field_data == None:
            for key, value in kwargs.items():
                kw_list.append(key)
                value_list.append(value)
            self._write_user_object_kw(class_type, field_name, field_default, kw_list, value_list, field_datatype)
        else:
            ck_req, miss_item = self._check_require(field_data, field_name, field_required)
            assert ck_req, 'The required data (%s) is missed in field_data'.format(miss_item)
            self._write_user_object_list(class_type, field_name, field_data, field_datatype)

    def _check_require(self, field_data, field_name, field_required):
        require_index = []
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
        # output = self._write_object(class_type, field_name, field_data)
        self._to_dic(class_type, field_data, field_datatype)

        # return output

    def _write_user_object_list(self, class_type, field_name, field_data, field_datatype):
        # output = self._write_object(class_type, field_name, field_data)
        self._to_dic(class_type, field_data, field_datatype)
        # return output
    
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
            self.idf_dic[class_type] = field_data
        
    def write_idf_file(self, file_path = os.getcwd()):
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

    def edit(self, class_type, class_name = '*', **kwargs):
        class_type = class_type.upper()
        class_idd = self.idd['properties'][class_type]
        field_name, _, field_default, field_required, field_datatype = self._get_idd_info(class_idd)

        field_data_list = self.idf_dic[class_type]

        for key, value in kwargs.items():
            # To write: check value type
            key = key.lower()
            index = field_name.index(key)
            if class_name == '*':
                for i in range(len(field_data_list)):
                    self.idf_dic[class_type][i][index] = value
            else:
                for i in range(len(field_data_list)):
                    if field_data_list[i][0] == class_name:
                        self.idf_dic[class_type][i][index] = value
                    else:
                        continue

    def options(self, class_type, att_name):
        pass




# data['properties']['Schedule:Compact']['legacy_idd']['field_info']
# data['properties']['Coil:Cooling:Water']['legacy_idd']['field_info']['water_inlet_node_name']
# # check_item:
# data['properties']['Fan:ConstantVolume']['patternProperties']['^.*\\S.*$']['properties']
# # check option:
# data['properties']['SurfaceConvectionAlgorithm:Inside']['patternProperties']['.*']['properties']['algorithm']['enum']
# # check_require
# data['properties']['Fan:ConstantVolume']['name']
# data['properties']['Fan:ConstantVolume']['patternProperties']['^.*\\S.*$']['required']
# # check_default:
# data['properties']['SurfaceConvectionAlgorithm:Inside']['patternProperties']['.*']['properties']['algorithm']['default']
# # check_max/min item num
# data['properties']['SurfaceConvectionAlgorithm:Inside']['maxProperties']
# data['properties']['Sizing:Zone']['min_fields']