import onnx 
import numpy as np
import configparser
import io
import os
from collections import defaultdict
import json
import logging
logging.basicConfig(
    level=logging.INFO, 
    format= '[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


class ModelParser():
    def __init__(self, model_cfg_path, max_memory=2*1024*1024):
        super(ModelParser, self).__init__()
        self.model_cfg_path = model_cfg_path
        logger.info("reading model description from {} ...".format(self.model_cfg_path))
        self.unique_config_file = self.unique_config_sections(model_cfg_path)
        self.cfg_reader = configparser.ConfigParser()
        self.max_memory = max_memory
        self.curr_memb = 0
        self.model_list = []
        self.parse()

    def unique_config_sections(self, config_file):
        """
        Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compatibility with configparser.
        Args:
            config_file:
        Returns:
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    def dtype_to_prec(self, dtype):
        if dtype == "fp32":
            prec = 32
        elif dtype[0:2]=="int" :
            prec = int(dtype[3:-1])
        else:
            return 8
        return prec

    def infer_activation_shape(self, wshape, ishape, padding, stride):
        # import ipdb as pdb; pdb.set_trace()
        b, iC, iH, iW = ishape
        ifC, fC, fH, fW = wshape
        assert (iC==ifC), logger.info("Input shape channel must match kernel input shape")
        oH=int((iH-fH+2*padding)/stride)+1
        oW=int((iW-fW+2*padding)/stride)+1
        oC=fC
        return [b, oC, oH, oW]

    def parse(self):
        self.cfg_reader.read_file(self.unique_config_file)
        idx = 0
        for section in self.cfg_reader.sections():
            logger.info('Parsing section {}'.format(section))
            if section.startswith('net'):
                batch = int(self.cfg_reader[section]['batch'])
                height = int(self.cfg_reader[section]['height'])
                width = int(self.cfg_reader[section]['width'])
                channels = int(self.cfg_reader[section]['channels'])
                layer = {"name": str(section),
                        "model": str(self.cfg_reader[section]['model']),
                        "type": "input",
                        "ishape": [batch, channels, height, width],
                        "oshape": [batch, channels, height, width],
                        "aprec": self.cfg_reader[section]['aprec'],
                        "id": idx}
                self.model_list.append(layer)
                idx += 1
            elif section.startswith('convolutional'):
                filters = int(self.cfg_reader[section]['filters'])
                size = int(self.cfg_reader[section]['size'])
                stride = int(self.cfg_reader[section]['stride'])
                padding = int(self.cfg_reader[section]['pad'])
                activation = self.cfg_reader[section]['activation']
                batch_normalize = 'batch_normalize' in self.cfg_reader[section]
                pad_type = 'same' if padding == 1 and stride == 1 else 'valid'
                ishape = self.model_list[idx-1]['oshape']
                wshape = [ishape[1], filters, size, size]
                oshape = self.infer_activation_shape(wshape, ishape, padding, stride)
                layer = {"name": str(section),
                        "type": "conv",
                         "channels": filters,
                         "wshape": wshape,
                         "pad_type": pad_type,
                         "pad": padding,
                         "oshape": oshape,
                         "activation": activation,
                         "aprec": self.cfg_reader[section]['aprec'],
                         "wprec": self.cfg_reader[section]['wprec'],
                         "id": idx}
                self.model_list.append(layer)
                idx += 1

    def get_conv_weight(self, node):
        pass

    def gen_tensor_by_type(self, shape, str_dtype, vals=None):
        tensor = []
        if str_dtype == "fp32":
            if vals != None:
                tensor = np.random.rand(shape).astype(np.float32)
            else:
                tensor = np.random.rand(*shape).astype(np.float32)
        elif str_dtype == "int32":
            if vals != None:
                tensor = np.zeros(shape).astype(np.int32)
            else:
                tensor = np.random.randint(2**32, size=shape).astype(np.int32)
        elif str_dtype == "int16":
            if vals != None:
                tensor = np.zeros(shape).astype(np.int16)
            else:
                tensor = np.random.randint(2**16, size=shape).astype(np.int16)
        elif str_dtype == "int8":
            if vals != None:
                tensor = np.zeros(shape).astype(np.int8)
            else:
                tensor = np.random.randint(2**8, size=shape).astype(np.int8)
        else:
            logger.info("{0} data type is not supported".format(str_dtype))
            exit()
        return tensor
        
    def print_model_graph(self):
        for layer in self.model_list:
            logger.info(json.dumps(layer, sort_keys=False, indent=4))
            # print(yaml.dump(layer, default_flow_style=True))

    def emit_8b(self, name, tensor, alignment='3'):
        array = tensor.flatten()
        code_str = ""
        code_str += (".global %s\n" % name)
        code_str += (".align " + alignment + "\n")
        code_str += ("%s:\n" % name)
        bs = array.tobytes()
        self.curr_memb += len(bs)
        if self.curr_memb > self.max_memory:
            logger.info("input model {} has excedded the available memory ({} Bytes): {} Bytes".format(self.model_cfg_path, self.max_memory, self.curr_memb))
        while ((len(bs) % 4) != 0):
            bs += bytes([0]) 
        for i in range(0, len(bs), 4):
            s = ""
            for n in range(4):
                s += "%02x" % bs[i+3-n] 
            code_str += ("    .word 0x%s\n" % s)
        return code_str

    def gen_code(self):
        code_str = ".section .data,\"aw\",@progbits\n\n"
        for layer in self.model_list:
            if layer['type'] == 'input':
                '''
                 For input layers, we only need to create input buffers
                '''
                # import ipdb as pdb; pdb.set_trace()
                itensor = self.gen_tensor_by_type(layer['ishape'], layer['aprec'])
                code_str += self.emit_8b("input_layer", itensor, 'NR_LANES*4') + "\n\n"
                logger.info("Generating data for {} of size {}".format("input_layer", itensor.shape))
            elif layer['type'] == 'conv':
                '''
                 For conv layer, we need to create:
                    - output buffers
                    - weight buffers
                '''
                # import ipdb as pdb; pdb.set_trace()
                wtensor = self.gen_tensor_by_type(layer['wshape'], layer['wprec'])
                code_str += self.emit_8b(layer['name']+"_wbuf", wtensor, 'NR_LANES*4')+ "\n\n"
                logger.info("Generating data for {} of size {}".format(layer['name']+"_wbuf", wtensor.shape))
                otensor = self.gen_tensor_by_type(layer['oshape'], layer['aprec'], vals=0)
                code_str += self.emit_8b(layer['name']+"_obuf", otensor, 'NR_LANES*4')+ "\n\n"
                logger.info("Generating data for {} of size {}".format(layer['name']+"_obuf", otensor.shape))

        out_file_name = self.model_list[0]['model'] + "_data.S"
        with open(out_file_name, "w") as f:
            f.write(code_str)
            f.close()
        