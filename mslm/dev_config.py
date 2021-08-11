import copy
import json

class devConfig():
    """
    Configuration for the dev files
    """
    def __init__(
        self,
        primary_dev_file=None,
        primary_dev_mode=None,
        bpc_secondary_dev_files=None,
        seg_secondary_dev_files=None,
        both_secondary_dev_files=None
    ):
        self.primary_dev_file = primary_dev_file
        self.primary_dev_mode = primary_dev_mode
        self.bpc_secondary_dev_files = bpc_secondary_dev_files
        self.seg_secondary_dev_files = seg_secondary_dev_files
        self.both_secondary_dev_files = both_secondary_dev_files

    @classmethod
    def from_dict(cls, dict_object):
        """
        Constructs a `devConfig` from a Python dictionary of parameters
        """
        config = devConfig()
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json(cls, json_file):
        """Constructs a `devConfig` from a json file of parameters"""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with open(json_file, "w") as writer:
            writer.write(self.to_json_string())
