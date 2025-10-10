import numpy as np

from napari_orientation.orientation_field_widget import (
    vector_field_widget,
    statistics_widget
)

def test_vector_field_widget(make_napari_viewer):

    viewer = make_napari_viewer()
    im_data = np.random.random((300, 300))
    layer = viewer.add_image(im_data)

    my_widget = vector_field_widget()

    vfield,_,layer_type = my_widget(viewer.layers[0])

    assert layer_type == 'vectors'


def test_statistics_widget(make_napari_viewer):
 
    viewer = make_napari_viewer()
    im_data = np.random.random((300, 300))
    layer = viewer.add_image(im_data)

    my_widget = statistics_widget(viewer=viewer)
    
    my_widget.single_frame.value = True
    my_widget._compute_colored_image()

    assert viewer.layers[1].data.shape == im_data.shape + (3,)
