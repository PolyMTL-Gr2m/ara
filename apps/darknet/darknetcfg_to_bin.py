import configparser
import argparse

sample_config = ""
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_cfg", required=True, help="input darknet cfg file")
    args = vars(ap.parse_args())
    config = configparser.ConfigParser(allow_no_value=True)
    print(config.read_file(args['input_cfg']))