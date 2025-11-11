# napari-orientation

[![License BSD-3](https://img.shields.io/pypi/l/napari-orientation.svg?color=green)](https://github.com/giocard/napari-orientation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-orientation.svg?color=green)](https://pypi.org/project/napari-orientation)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-orientation.svg?color=green)](https://python.org)
[![tests](https://github.com/giocard/napari-orientation/workflows/tests/badge.svg)](https://github.com/giocard/napari-orientation/actions)
[![codecov](https://codecov.io/gh/giocard/napari-orientation/branch/main/graph/badge.svg)](https://codecov.io/gh/giocard/napari-orientation)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-orientation)](https://napari-hub.org/plugins/napari-orientation)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A napari plugin to analyse local orientation in images.

## Installation

You can install the plugin from the napari GUI interface by going to ```Plugins/Install\Uninstall Plugins``` and selecting `napari-orientation`.
Alternatively, you can install the plugin from the napari conda environment via [pip]:

```text
pip install napari-orientation
```

## Usage

You can access all the functionalities of the plugin from the menu ```Plugins\Orientation Analysis```.

All the analyses work only on single-channel 2D images and on single-channel 2D time series.
In this last case the analysis can be restricted to single frames.

The only parameter available is the sigma smoothing, in pixels, which controls the strength of the gaussian filter applied to the gradient of the image before computing the orientation vector pixelwise.

### Compute orientation metrics

This GUI gives access to most of the functionalities. You can compute several metrics and display them as images.

Definitions:

* __Orientation vector__: unitary vector describing the average orientation of the gradient of the intensity around a point in the image. Computationally this is obtained by determining the first eigenvector (i.e. largest eigenvalue) of the structure tensor matrix.
* __Orientation field__: mapping of the orientation vectors at each point in the image.
* __Angle field__: mapping of the orientation angle at each point in the image
* __Energy__: sum of the tensor eigenvalues.
* __Coherence__: ratio between the difference and the sum of the maximum and minimum tensor eigenvalues.
* __Curvature__: rate of change in the local orientation in the direction perpendicular to that orientation.
* __Correlation length__: distance where the radial autocorrelation of the angle field drops below 0.5

#### Display Colored Orientation

It computes an image where each pixel is colored differently according to the orientation angle estimated at that position.

![Example colored orientation](docs/example_colored_orientation.png)

#### Display Coherence

It computes an image where the value of each pixel represents the coherence estimated at that position.

![Example coherence](docs/example_coherence.png)

#### Display Curvature

It computes an image where the value of each pixel represents the curvature estimated at that position.

![Example coherence](docs/example_curvature.png)

#### Display Angle

It computes an image where the value of each pixel represents the angle, in degrees, estimated at that position.

![Example coherence](docs/example_angle.png)

#### Compute statistics

Estimate the average value for the following metrics: Energy, Coherence, Correlation length, Curvature.

Note that Correlation length and Curvature metrics are both provided in physical units, and therefore their determination relies on the accuracy of the pixel size provided for the image. Using the napari-bioformats to open the images should guarantee that the pixel size stored in the file metadata is properly read by the plugin. In any case it always possible to adjust the pixel size from the interface.

![Example statistics](docs/example_statistics.png)

### Generate vector-coded images

It generates a vector layer displaying the orientation field estimated locally, over a grid with spacing defined by the user.

![Example vectors](docs/example_vectors.png)

This is a separate widget because it currently only works on single frames.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-orientation" is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.

## Credits

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

This work was inspired by the plugin [OrientationJ] for ImageJ, that was partially converted by the same developers into a [plugin for napari](https://github.com/EPFL-Center-for-Imaging/napari-orientationpy).

napari-orientation focuses on the computation of several metrics, some of them proposed in this [article](https://doi.org/10.1038/s41467-019-13702-4)

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[OrientationJ]: https://github.com/Biomedical-Imaging-Group/OrientationJ
