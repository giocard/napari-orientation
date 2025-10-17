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

```
pip install napari-orientation
```

## Usage

You can access all the functionalities of the plugin from the menu ```Plugins\Orientation Analysis```.

All the analyses work only on single-channel 2D images and on single-channel 2D time series.
In this last case the analysis can be restricted to single frames.

The only parameter available is the sigma smoothing, in pixels, which controls the strength of the gaussian filter applied to the gradient of the image before computing the orientation vector pixelwise.

### Compute orientation metrics

This GUI gives access to most of the functionalities. You can compute several metrics and display them as images.

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

Estimate the average value for the following metrics: Energy, Coherence, Correlation length, Curvature. The curvature is the only metric in physical units, and therefore the determination of its values relies on the accuracy of the pixel size provided for the image.

![Example statistics](docs/example_statistics.png)

### Compute orientation metrics


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
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[OrientationJ]: https://github.com/Biomedical-Imaging-Group/OrientationJ
