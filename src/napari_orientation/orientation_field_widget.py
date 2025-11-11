from typing import TYPE_CHECKING

import napari.types
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Button, Container, Table, create_widget
from napari.layers import Image

if TYPE_CHECKING:
    import napari


@magic_factory(
    img_layer={"label": "Image"},
    call_button="Generate orientation vector field",
)
def vector_field_widget(
    img_layer: Image,
    Sigma_smoothing: int = 4,  # Gaussian smoothing sigma for structure tensor
    Step: int = 10,  # subsampling step for vector field
) -> napari.types.LayerDataTuple:
    from napari.utils.notifications import show_info

    input_image = img_layer.data
    if len(input_image.shape) > 3:
        show_info(
            f"Input image must be single-channel image or 2D time series. This image has {len(input_image.shape)} dimensions"
        )
        return
    if input_image.ndim == 3 and input_image.shape[-1] <= 3:
        show_info(
            "Input image appears to be a RGB image. Please provide a single-channel image or time series."
        )
        return

    slice_pfx = ""
    if input_image.ndim == 3:
        this_slice = int(img_layer._slice.slice_input.world_slice.point[0])
        input_image = input_image[this_slice]
        slice_pfx = f"_slice{this_slice}"

    orientation_field, eigenval_field = compute_orientation_field(
        input_image, sigma=Sigma_smoothing
    )

    offset = Step // 2
    # Subsample orientation field for vector display
    sampled_field = orientation_field[offset::Step, offset::Step, :]
 
    usub = sampled_field[:, :, 0]
    vsub = -sampled_field[:, :, 1]
 
    vectors_field = np.stack([usub,vsub], axis=-1)

    # make sure grid is off
    #viewer = napari.current_viewer()
    #viewer.grid.enabled = False
 
    return (
        vectors_field,
        {
            "name": f"{img_layer.name}{slice_pfx}_vectors_σ={Sigma_smoothing:.1f}",
            "scale": [Step, Step],
            "translate": [offset, offset],
            "edge_width": 0.2,
            "length": 0.7,
            "vector_style": "line",
        },
        "vectors",
    )


class statistics_widget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self._viewer = viewer

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self.pixel_size = create_widget(
            label="Pixel size (\u03bcm)", annotation=float, options={'value': 1.0, 'step': 0.0001}
        )

        self.single_frame = create_widget(
            label="Only visible frame",
            annotation=bool,
            options={"value": True},
        )

        self.sigma = create_widget(
            label="Sigma smoothing", annotation=int, options={'value': 4}
        )

        self.btn_color = Button(text="Display Colored Orientation")
        self.btn_color.clicked.connect(self._compute_colored_image)

        self.btn_coherence = Button(text="Display Coherence")
        self.btn_coherence.clicked.connect(self._compute_coherence_image)

        self.btn_curvature = Button(text="Display Curvature")
        self.btn_curvature.clicked.connect(self._compute_curvature_image)

        self.btn_angle = Button(text="Display Angle")
        self.btn_angle.clicked.connect(self._compute_angle_image)

        self.btn_stats = Button(text="Compute statistics")
        self.btn_stats.clicked.connect(self._compute_statistics)

        self.btn_savetab = Button(text="Save Table as csv...")
        self.btn_savetab.clicked.connect(self._save_table)

        self.table_data = [
            {
                "Frame": "",
                "Energy": "",
                "Coherence": "",
                "Correlation": "",
                "Curvature": "",
            }
        ]
        self.table = Table(self.table_data)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._image_layer_combo,
                self.pixel_size,
                self.single_frame,
                self.sigma,
                self.btn_color,
                self.btn_coherence,
                self.btn_curvature,
                self.btn_angle,
                self.btn_stats,
                self.table,
                self.btn_savetab,
            ]
        )

        # Connect the change event
        self._image_layer_combo.changed.connect(self._on_image_layer_change)

        # Set initial value according to current selection
        self._on_image_layer_change(self._image_layer_combo.value)

        return

    def _on_image_layer_change(self, value):
        """Update pixel_size when the selected image layer changes."""
        image_layer = value
        if image_layer is not None and hasattr(image_layer, 'scale'):
            # Typically, image_layer.scale is a tuple of floats representing pixel size in each dimension
            self.pixel_size.value = float(image_layer.scale[-1])
        else:
            # Fallback value if no image_layer
            self.pixel_size.value = 1.0

    def _compute_colored_image(self):
        from napari.utils import cancelable_progress
        from napari.utils.notifications import show_info

        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        input_image = image_layer.data
        sigma = self.sigma.value
        only_current_slice = self.single_frame.value

        if not is_image_valid(input_image):
            return

        input_image, twod_analysis, this_slice, slice_pfx = extract_image(
            input_image, only_current_slice, image_layer
        )

        # If the image is 2D, process as a single slice
        if twod_analysis:
            angle_map_rgb = compute_color_image(input_image, sigma=sigma)
        # If the image is 3D, process slice by slice along the first axis
        else:
            angle_map_rgb = np.empty(input_image.shape + (3,), dtype=np.uint8)
            with cancelable_progress(
                input_image, miniters=5
            ) as pbr:  # , cancel_callback=cancel_callback) as pbr:
                for i, im_slice in enumerate(pbr):
                    angle_map_rgb[i] = compute_color_image(
                        im_slice, sigma=sigma
                    )
                if pbr.is_canceled:
                    show_info("Operation canceled - no image generated!")
                    del angle_map_rgb
                    return None

        layer = self._viewer.add_image(
            angle_map_rgb,
            name=f"{image_layer.name}{slice_pfx}_colored_σ={sigma:.1f}",
            rgb=True,
            colormap="hsv",
        )

        # make colorbar visible in this case
        #layer.colorbar.visible = True
 
        # generate custom colorbar for orientation
        def make_orientation_colorbar(viewer=self._viewer):
            from qtpy.QtWidgets import QVBoxLayout, QWidget
            from vispy import scene
            import vispy.color as vpcolor
            from vispy.scene import widgets

            for dwk in self._viewer.window.dock_widgets.keys():
                if dwk == 'Orientation angle (degrees)':
                    return

            cbCanvas = scene.SceneCanvas()
            cbCmap = vpcolor.get_colormap('hsv')
            cbColorbar = widgets.ColorBarWidget(cbCmap, 'right', "", 'white',
                    clim=(-90, 90),
                    border_width=1.,
#                    border_color='yellow',
                    padding=(.4, .4),
                    axis_ratio=0.1
                    )
#            cbColorbar.ticks = ['-90', '-45', '0', '45', '90']
            cbCanvas.central_widget.add_widget(cbColorbar)

            layout = QVBoxLayout()
            cb_widget = QWidget()
            layout.addWidget(cbCanvas.native)
            cb_widget.setLayout(layout)
            cb_widget.setMinimumHeight(200)

            viewer.window.add_dock_widget(cb_widget, name='Orientation angle (degrees)', area = 'right')

            return
        
        cb_widget = make_orientation_colorbar()

        return

    def _compute_coherence_image(self):
        from napari.utils import cancelable_progress
        from napari.utils.notifications import show_info

        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        input_image = image_layer.data
        sigma = self.sigma.value
        only_current_slice = self.single_frame.value

        if not is_image_valid(input_image):
            return

        input_image, twod_analysis, this_slice, slice_pfx = extract_image(
            input_image, only_current_slice, image_layer
        )

        # If the image is 2D, process as a single slice
        if twod_analysis:
            coherence_map = compute_coherence_image(input_image, sigma=sigma)
        # If the image is 3D, process slice by slice along the first axis
        else:
            coherence_map = np.empty(input_image.shape, dtype=np.float32)
            with cancelable_progress(
                input_image, miniters=5
            ) as pbr:  # , cancel_callback=cancel_callback) as pbr:
                for i, im_slice in enumerate(pbr):
                    coherence_map[i] = compute_coherence_image(
                        im_slice, sigma=sigma
                    )
                if pbr.is_canceled:
                    show_info("Operation canceled - no image generated!")
                    del coherence_map
                    return None

        self._viewer.add_image(
            coherence_map,
            name=f"{image_layer.name}{slice_pfx}_coherence_σ={sigma:.1f}",
            colormap="magma",
        )

        return

    def _compute_curvature_image(self):
        from napari.utils import cancelable_progress
        from napari.utils.notifications import show_info

        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        print(image_layer.scale[-1])
        input_image = image_layer.data
        sigma = self.sigma.value
        only_current_slice = self.single_frame.value
        pixel_size = self.pixel_size.value

        if not is_image_valid(input_image):
            return

        input_image, twod_analysis, this_slice, slice_pfx = extract_image(
            input_image, only_current_slice, image_layer
        )

        # If the image is 2D, process as a single slice
        if twod_analysis:
            curvature_map = compute_curvature_image(
                input_image, pixel=pixel_size, sigma=sigma
            )
        # If the image is 3D, process slice by slice along the first axis
        else:
            curvature_map = np.empty(input_image.shape, dtype=np.float32)
            with cancelable_progress(
                input_image, miniters=5
            ) as pbr:  # , cancel_callback=cancel_callback) as pbr:
                for i, im_slice in enumerate(pbr):
                    curvature_map[i] = compute_curvature_image(
                        im_slice, sigma=sigma
                    )
                if pbr.is_canceled:
                    show_info("Operation canceled - no image generated!")
                    del curvature_map
                    return None

        self._viewer.add_image(
            curvature_map,
            name=f"{image_layer.name}{slice_pfx}_curvature_σ={sigma:.1f}",
            colormap="gray_r",
        )

        return

    def _compute_angle_image(self):
        from napari.utils import cancelable_progress
        from napari.utils.notifications import show_info

        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        input_image = image_layer.data
        sigma = self.sigma.value
        only_current_slice = self.single_frame.value

        if not is_image_valid(input_image):
            return

        input_image, twod_analysis, this_slice, slice_pfx = extract_image(
            input_image, only_current_slice, image_layer
        )

        # If the image is 2D, process as a single slice
        if twod_analysis:
            angle_map = compute_angle_image(input_image, sigma=sigma)
        # If the image is 3D, process slice by slice along the first axis
        else:
            angle_map = np.empty(input_image.shape, dtype=np.float32)
            with cancelable_progress(
                input_image, miniters=5
            ) as pbr:  # , cancel_callback=cancel_callback) as pbr:
                for i, im_slice in enumerate(pbr):
                    angle_map[i] = compute_angle_image(im_slice, sigma=sigma)
                if pbr.is_canceled:
                    show_info("Operation canceled - no image generated!")
                    del angle_map
                    return None

        self._viewer.add_image(
            angle_map,
            name=f"{image_layer.name}{slice_pfx}_angle_σ={sigma:.1f}",
            colormap="gray",
        )

        return

    def _save_table(self):
        from napari.utils.notifications import show_info
        from qtpy.QtWidgets import QFileDialog

        # test if empty either before or after the conversion to dataframe
        if self.table.shape[0] < 1 or (
            self.table.shape[0] == 1 and not self.table["Frame"][0]
        ):
            show_info("No data in the table to save!")
            return

        df = self.table.to_dataframe()

        # Initialize and launch GUI
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getSaveFileName(
            None,
            "Save table as...",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if filePath:
            if not filePath.lower().endswith(".csv"):
                filePath += ".csv"
            try:
                df.to_csv(filePath, index=False)
                show_info(f"Table saved to {filePath}")
            except:
                show_info(f"Failed to save table to {filePath}")
                raise
        return

    def _compute_statistics(self):
        from napari.utils import cancelable_progress
        from napari.utils.notifications import show_info

        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        input_image = image_layer.data
        sigma = self.sigma.value
        only_current_slice = self.single_frame.value
        pixel_size = self.pixel_size.value

        if not is_image_valid(input_image):
            return

        input_image, twod_analysis, this_slice, slice_pfx = extract_image(
            input_image, only_current_slice, image_layer
        )

        # If the image is 2D, process as a single slice
        if twod_analysis:
            om = compute_image_orientation_statistics(
                input_image, this_slice, pixel_size, sigma
            )
            omdict = om._asdict()
            tabdata = [omdict]
            self.table.set_value(tabdata)
        # If the image is 3D, process slice by slice along the first axis
        else:
            with cancelable_progress(
                input_image, miniters=5
            ) as pbr:  # , cancel_callback=cancel_callback) as pbr:
                timedata = []
                for i, im_slice in enumerate(pbr):
                    # pbr.set_description(f'Slice {i+1}')
                    om = compute_image_orientation_statistics(
                        im_slice, i + 1, pixel_size, sigma
                    )
                    timedata.append(om._asdict())
                    self.table.set_value(timedata)
            if pbr.is_canceled:
                show_info("Operation canceled!")
                return None

        return


#################################################################################################################


def is_image_valid(input_image):
    from napari.utils.notifications import show_info

    if len(input_image.shape) > 3:
        show_info(
            f"Input image must be single-channel image or 2D time series. This image has {len(input_image.shape)} dimensions"
        )
        return False
    if input_image.ndim == 3 and input_image.shape[-1] <= 3:
        show_info(
            "Input image appears to be a RGB image. Please provide a single-channel image or time series."
        )
        return False

    return True


def extract_image(input_image, only_current_slice, image_layer):
    slice_pfx = ""
    if input_image.ndim == 2:
        twod_analysis = True
        this_slice = 1
    elif input_image.ndim == 3 and only_current_slice:
        islice = int(image_layer._slice.slice_input.world_slice.point[0])
        input_image = input_image[islice]
        this_slice = islice + 1
        slice_pfx = f"_slice{this_slice}"
        twod_analysis = True
    else:
        twod_analysis = False
        this_slice = -1

    return input_image, twod_analysis, this_slice, slice_pfx


def compute_image_orientation_statistics(
    inimage, this_slice, pixel_size, sigma=4
):
    from collections import namedtuple

    fields = ["Frame", "Energy", "Coherence", "Correlation", "Curvature"]
    Measures = namedtuple("Measures", fields)

    orientation_field, eigenval_field = compute_orientation_field(
        inimage, sigma=sigma
    )
    coherence_map = compute_coherence(eigenval_field)
    curvature_map = compute_curvature(orientation_field, pixel=pixel_size)
    mean_curvature, fit_params, curvature_distribution = (
        fit_curvature_distribution(curvature_map)
    )
    energy_map = compute_energy(eigenval_field)
    correlation_map = compute_orientation_correlation(orientation_field)
    correlation_curve, correlation_radius = radial_average(
        correlation_map, pixel_size
    )
    correlation_length = half_max_position(
        correlation_curve, correlation_radius
    )

    om = Measures(
        Frame=f"{this_slice}",
        Energy=f"{np.mean(energy_map)}",
        Curvature=f"{mean_curvature}",
        Coherence=f"{np.mean(coherence_map)}",
        Correlation=f"{correlation_length}",
    )

    return om


def compute_color_image(inimage, sigma=4):

    orientation_field, eigenval_field = compute_orientation_field(
        inimage, sigma=sigma
    )
    angle_map = compute_angle(orientation_field)
    coherency_map = compute_coherence(eigenval_field)

    angle_map_rgb = generate_colored_orientation_map(
        angle_map, inimage, coherency_map
    )

    return angle_map_rgb


def compute_angle_image(inimage, sigma=4):

    orientation_field, eigenval_field = compute_orientation_field(
        inimage, sigma=sigma
    )
    angle_map = compute_angle(orientation_field)
    # del orientation_field, eigenval_field

    return angle_map


def compute_coherence_image(inimage, sigma=4):

    orientation_field, eigenval_field = compute_orientation_field(
        inimage, sigma=sigma
    )
    coherency_map = compute_coherence(eigenval_field)

    return coherency_map


def compute_curvature_image(inimage, pixel=1, sigma=0):
    # based on "Curvature Estimation from Orientation Fields" by M. van Ginkel, 1999
    # based on formulas 4, 6 and 8 in the paper

    # compute orientation field
    orientation_field, eigenval_field = compute_orientation_field(
        inimage, sigma=sigma
    )

    # compute curvature from orientation field
    curvature_map = compute_curvature(orientation_field, pixel=pixel)

    return curvature_map


def fit_curvature_distribution(curvature_map):
    from scipy.optimize import curve_fit

    def exp1_decay(x, a):
        return a * np.exp(-a * x)

    # Flatten and take absolute values
    data = np.abs(curvature_map.flatten())
    data = data[data > 0]  # Remove zeros

    # Normalized Histogram
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    curvature_distribution = {
        "bin_centers": bin_centers,
        "hist": hist,
        "bin_edges": bin_edges,
    }

    # Fit exponential decay
    popt, _ = curve_fit(exp1_decay, bin_centers, hist, p0=(1))

    # Mean half-life of decay is ln(2)/b
    mean_curvature = np.log(2) / popt[0]

    return mean_curvature, popt, curvature_distribution


def compute_orientation_field(inimage, sigma=4):
    from scipy.ndimage import gaussian_filter, median_filter

    # Remove hot pixels
    image = median_filter(
        inimage, size=3
    )  # Apply median filter to remove hot pixels

    # Compute image gradients
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)

    # Compute elements of the structure tensor
    Ixx = gaussian_filter(Ix * Ix, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma=sigma)

    # Stack components into a (H, W, 2, 2) tensor
    H, W = image.shape
    J = np.empty((H, W, 2, 2))
    J[:, :, 0, 0] = Ixx
    J[:, :, 0, 1] = Ixy
    J[:, :, 1, 0] = Ixy
    J[:, :, 1, 1] = Iyy

    # Reshape for batch eigen-decomposition: (H*W, 2, 2)
    J_reshape = J.reshape(-1, 2, 2)
    vals, vecs = np.linalg.eigh(J_reshape)

    # Dominant eigenvectors (corresponding to largest eigenvalue)
    dominant_vecs = vecs[:, :, 1]
    dominant_vecs /= (
        np.linalg.norm(dominant_vecs, axis=1, keepdims=True) + 0.0001
    )  # Normalize to avoid division by zero

    # Reshape back to image grid
    orientation_field = dominant_vecs.reshape(H, W, 2)
    eigenval_field = vals.reshape(H, W, 2)

    return orientation_field, eigenval_field


def vector_angle(vector):
    """Calculates the angle (in radians) of a 2D array."""
    return np.arctan2(vector[:, :, 0], vector[:, :, 1])


def compute_angle(orientation_field):
    """Compute angle map in degrees from orientation field, constrained to [-90, 90] degrees range."""
    angleMap = vector_angle(orientation_field)
    # convert [-pi,pi] to [-pi/2, pi/2]
    angleMap = np.where(angleMap < -0.5 * np.pi, angleMap + np.pi, angleMap)
    angleMap = np.where(angleMap > 0.5 * np.pi, angleMap - np.pi, angleMap)
    # convert radians to degrees
    angleMap = np.degrees(angleMap)
    return angleMap


def compute_coherence(eigenval_field):
    # Compute coherency map from eigenvalue field
    coherency = (eigenval_field[:, :, 1] - eigenval_field[:, :, 0]) / (
        np.maximum(eigenval_field[:, :, 0] + eigenval_field[:, :, 1], 0.0001)
    )
    return coherency


def compute_curvature(orientation_field, pixel=1):
    ocopy = orientation_field.copy()
    alfa = vector_angle(ocopy)
    beta = np.exp(2j * alfa)

    betadx = np.gradient(beta, axis=1)
    betady = np.gradient(beta, axis=0)

    dx = -1j / (2.0 * beta) * betadx / pixel
    dy = -1j / (2.0 * beta) * betady / pixel

    # Perpendicular vector to orientation (u_perp = [-uy, ux])
    u_perp_x = -orientation_field[:, :, 0]
    u_perp_y = orientation_field[:, :, 1]

    # Compute curvature
    # we are not interested to the sign of curvature, only the magnitude
    curvature_map = np.abs(np.real(dx * u_perp_x + dy * u_perp_y))

    return curvature_map


def compute_energy(eigenval_field):
    # Compute energy map from eigenvalue field
    energy = np.sum(np.abs(eigenval_field), axis=-1)
    return energy


def generate_colored_orientation_map(angle_map, image, coherency_map):
    from skimage.color import hsv2rgb

    # Generate orientation angle map as Hue color overlaid on image
    # normalize to [0, 1]
    angle_map_normalized = (angle_map + 90) / 180.0  # Map [-90, 90] to [0, 1]
    # create HSV image
    hsvImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    hsvImage[..., 0] = angle_map_normalized  # Hue
    hsvImage[..., 1] = coherency_map  # 1.0  # Saturation
    hsvImage[..., 2] = (image - np.min(image)) / (
        np.max(image) - np.min(image)
    )  # Value
    # convert HSV to RGB
    angle_map_rgb = hsv2rgb(hsvImage)
    angle_map_rgb = (angle_map_rgb * 255).astype(np.uint8)

    return angle_map_rgb


def compute_orientation_correlation(orientation_field):
    from numpy.fft import fft2, ifft2

    # Calculate orientation angles and phasors
    alfa = np.arctan2(orientation_field[:, :, 0], orientation_field[:, :, 1])
    beta = np.exp(2j * alfa)

    # Compute FFT
    F_beta = fft2(beta)
    # Autocorrelation using FFT: inverse FFT of the power spectrum
    orientationcorr = ifft2(F_beta * np.conj(F_beta))
    # Center the autocorrelation map
    # np.fft.fftshift centers the zero frequency component in the output
    orientationcorr = np.fft.fftshift(orientationcorr)
    orientationcorr = np.real(orientationcorr)

    # Normalize
    orientationcorr /= np.max(np.abs(orientationcorr))

    return orientationcorr


def radial_average(data, pixelsize, center=None):
    # Default to the center of the array
    if center is None:
        center = (data.shape[0] // 2, data.shape[1] // 2)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    # Aggregate mean by radius
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    # Create radial axis in micrometers
    radial_axis = (
        np.arange(len(radialprofile)) * pixelsize
    )  # Convert pixel indices
    radial_axis = radial_axis[
        : len(radialprofile)
    ]  # Ensure radial_axis matches the length of radialprofile

    return radialprofile, radial_axis


def half_max_position(radial_profile, radial_axis):
    max_val = np.max(radial_profile)
    half_max = max_val / 2
    for i, val in enumerate(radial_profile):
        if val <= half_max:
            xval = radial_axis[i - 1] + (
                radial_axis[i] - radial_axis[i - 1]
            ) * (half_max - radial_profile[i - 1]) / (
                radial_profile[i] - radial_profile[i - 1]
            )
            return xval  # Linear interpolation for better accuracy
    # If no value drops below half max, return 0
    return 0
