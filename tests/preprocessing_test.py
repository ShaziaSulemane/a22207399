import os

import tensorflow as tf

from image_network_library import preprocessing as rd

pwd_path = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_json_read():
    path = "/via_project_12Dec2022_19h12m_json.json"
    mydict = rd.read_json(pwd_path + path, verbose=0)
    assert len(mydict) == 2


def test_find_floor():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]]}
    floor_dict = rd.find_floor(mydict)
    assert len(mydict) == 2
    assert floor_dict.get('frame8.jpg') == 603
    assert floor_dict.get('frame7.jpg') == 603


def test_measure_length_avg():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}
    distances, avg = rd.measure_length(mydict, [[0, 1], [1, 2]], mode="avg", verbose=0)
    assert len(distances) == 3
    assert len(avg) == 2


def test_measure_length_median():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}
    distances, m = rd.measure_length(mydict, [[0, 1], [1, 2]], mode="median", verbose=0)
    assert len(distances) == 3
    assert len(m) == 2


def test_measure_length_none():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}
    distances, m = rd.measure_length(mydict, [[0, 1], [1, 2]], verbose=0)
    assert len(distances) == 3
    assert m is None


def test_measure_length_none_pixel_irl():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}
    distances, m = rd.measure_length(mydict, [[0, 1], [1, 2]], verbose=0, pixel_irl=0.003)
    assert len(distances) == 3
    assert m is None


def test_make_tensors():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}
    tensors = rd.make_tensors(mydict)
    assert len(tensors) == 3
    assert tensors['frame8.jpg'].dtype == tf.int32
    assert tensors['frame7.jpg'].dtype == tf.int32
    assert tensors['frame6.jpg'].dtype == tf.int32
    assert tensors['frame8.jpg'].shape == (3, 2)
    assert tensors['frame7.jpg'].shape == (3, 2)
    assert tensors['frame6.jpg'].shape == (3, 2)
