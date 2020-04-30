import configparser, os


def create_dict(file):
    """
    :param file: cfg file with the parameters
    :return: dictionary with the important settings
    """
    settings_dict = {}
    line_index = 0
    line = file.readline()
    while line:
        line_index += 1
        if line_index == 5:
            settings_dict["Optimization"] = line[20:-3]
        if line_index == 7:
            settings_dict["Contrast"] = int(line[14:-2])
        if line_index == 9:
            settings_dict["Gain"] = int(line[10:-2])
        if line_index == 17:
            settings_dict["FocusDepth"] = float(line[16:-2])
        line = file.readline()
    return settings_dict


if __name__ == "__main__":
    file_name = 'ScanningCurrent Prostate (Autoscan (m4DC7-3 40)).cfg'
    file = open(file_name)
    config = configparser.ConfigParser()
    settings_dict = create_dict(file)

