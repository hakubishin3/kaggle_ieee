import json
import codecs


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config, save_path):
    """save json-file"""
    f = codecs.open(save_path, 'w', 'utf-8')
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
