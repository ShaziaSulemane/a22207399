import unittest
from image_network_library import preprocessing as rd
import tracemalloc

tracemalloc.start()


class preprocessing_test(unittest.TestCase):
    def test_json_read(self):
        path = "via_project_12Dec2022_19h12m_json.json"
        mydict = rd.read_json(path, verbose=0)
        assert len(mydict) == 2

    def test_find_floor(self):
        path = "via_project_12Dec2022_19h12m_json.json"
        mydict = rd.read_json(path, verbose=0)
        floor_dict = rd.find_floor(mydict)
        assert len(mydict) == 2

    def test_measure_lenght(self):
        path = "via_project_12Dec2022_19h12m_json.json"
        mydict = rd.read_json(path, verbose=0)
        distances, avg = rd.measure_length(mydict, [[0, 1], [1, 2]], mode='avg')
        assert len(distances) == 2 and len(avg) == 2


if __name__ == '__main__':
    unittest.main()
