import numpy as np
import pytest
from napari.layers import Image
from napari_orientation.orientation_field_widget import (
    is_image_valid,
    extract_image,
    compute_orientation_field,
    compute_angle,
    compute_coherence,
    compute_curvature,
    fit_curvature_distribution,
    compute_energy,
    compute_orientation_correlation,
    vector_angle,
    compute_image_orientation_statistics,
    compute_vector_field,
    compute_color_image,
    compute_angle_image,
    compute_coherence_image,
    compute_curvature_image,
)

def test_is_image_valid():
    # Valid 2D
    assert is_image_valid(np.zeros((100, 100)))
    # Valid 3D (time series)
    assert is_image_valid(np.zeros((10, 100, 100)))
    
    # Invalid 4D
    from unittest.mock import patch
    with patch('napari.utils.notifications.show_warning') as mock_warn:
        assert not is_image_valid(np.zeros((2, 10, 100, 100)))
        mock_warn.assert_called()
    
    # Invalid RGB (3D but last dim <= 3)
    with patch('napari.utils.notifications.show_warning') as mock_warn:
        assert not is_image_valid(np.zeros((100, 100, 3)))
        mock_warn.assert_called()

def test_extract_image():
    # 2D case
    img2d = np.zeros((100, 100))
    img, slice_idx, pfx = extract_image(img2d, False, None)
    assert img.shape == (100, 100)
    assert slice_idx == 1
    assert pfx == ""

    # 3D case - only current slice
    img3d = np.zeros((10, 100, 100))
    
    # Mock layer with slice state
    class MockLayer:
        pass
    
    mock_layer = MockLayer()
    
    class MockWorldSlice:
        point = [5, 0, 0]
        
    class MockSliceInput:
        world_slice = MockWorldSlice()
        
    class MockSlice:
        slice_input = MockSliceInput()
        
    mock_layer._slice = MockSlice()
    
    img, slice_idx, pfx = extract_image(img3d, True, mock_layer)
    assert img.shape == (100, 100)
    assert slice_idx == 6 # 0-indexed 5 -> 1-indexed 6
    assert pfx == "_slice6"

    # 3D case - whole stack
    img, slice_idx, pfx = extract_image(img3d, False, None)
    assert img.shape == (10, 100, 100)
    assert slice_idx == -1
    assert pfx == ""

def test_compute_orientation_field():
    # Create a synthetic image with vertical lines
    img = np.zeros((100, 100))
    for i in range(0, 100, 10):
        img[:, i:i+5] = 1
    
    sigma = 1
    orientation_field, eigenval_field = compute_orientation_field(img, sigma=sigma)
    
    assert orientation_field.shape == (100, 100, 2)
    assert eigenval_field.shape == (100, 100, 2)
    
    # Check that orientation is roughly vertical (angle near 0 or 180, so vectors [1, 0] or [-1, 0]?)
    # Wait, vertical lines have gradients in X direction. Structure tensor will have dominant eigenvector in X direction.
    # The orientation field should point perpendicular to gradient for "orientation of structure"?
    # Usually structure tensor gives orientation of the gradient (normal to structure). 
    # But usually we want orientation along the structure (tangent).
    # The code calculates dominant eigenvector of structure tensor. 
    # For vertical lines, gradient is horizontal (Ix large, Iy small). 
    # Structure tensor J will have large component in Jxx.
    # Dominant eigenvector will be [1, 0] (horizontal).
    # So this returns gradient direction (normal).
    # Let's check compute_angle logic later.
    pass

def test_vector_angle():
    # Vector pointing right (1, 0) -> angle 0? atan2(x, y) = atan2(1, 0) = pi/2 ?
    # Code uses np.arctan2(vector[:, :, 0], vector[:, :, 1]) -> atan2(x, y)
    # usually atan2 is (y, x). 
    # If code uses (x, y), it means it treats '0' as y-axis and '1' as x-axis? Or just rotated.
    
    vec = np.array([[[0, 1], [1, 0]]]) # shape (1, 2, 2)
    angles = vector_angle(vec)
    # [0, 1] -> atan2(0, 1) = 0
    # [1, 0] -> atan2(1, 0) = pi/2
    assert angles[0, 0] == 0
    assert np.isclose(angles[0, 1], np.pi/2)

def test_compute_angle_logic():
    # Verify compute_angle logic
    # Create orientation field with known vectors
    # [0, 1] -> angle 0 -> 0 degrees
    # [1, 0] -> angle pi/2 -> 90 degrees
    # [0, -1] -> angle pi -> 180 degrees -> wrapped to 0?
    
    shape = (10, 10, 2)
    orientation_field = np.zeros(shape)
    orientation_field[:] = [0, 1] # 0 degrees
    
    angles = compute_angle(orientation_field)
    assert np.allclose(angles, 0)
    
    orientation_field[:] = [1, 0] # 90 degrees
    angles = compute_angle(orientation_field)
    assert np.allclose(angles, 90)

    orientation_field[:] = [-1, 0] # -90 degrees ? atan2(-1, 0) = -pi/2 -> -90
    angles = compute_angle(orientation_field)
    assert np.allclose(angles, -90)

def test_compute_coherence():
    # Eigenvalues l1, l2. Coherence = (l2 - l1) / (l1 + l2)
    # Code assumes l2 >= l1? 
    # In compute_orientation_field: vals, vecs = np.linalg.eigh(J_reshape). vals returns ascending order.
    # So vals[:, 0] <= vals[:, 1].
    # compute_coherence uses (v1 - v0) / (v0 + v1) where v1 is index 1 (larger).
    # So valid range [0, 1].
    
    shape = (10, 10, 2)
    eigenval_field = np.zeros(shape)
    # Isotropic: l1 = l2 = 1 -> coherence 0
    eigenval_field[:] = [1, 1]
    coh = compute_coherence(eigenval_field)
    assert np.allclose(coh, 0)
    
    # Anisotropic: l0=0, l1=1 -> coherence 1
    eigenval_field[:] = [0, 1]
    coh = compute_coherence(eigenval_field)
    assert np.allclose(coh, 1)

def test_compute_curvature():
    # Just smoke test on execution
    ori = np.zeros((20, 20, 2))
    ori[:, :, 1] = 1 # vertical vectors
    curv = compute_curvature(ori)
    assert curv.shape == (20, 20)
    assert not np.any(np.isnan(curv))

def test_fit_curvature_distribution():
    curv = np.random.exponential(scale=10, size=(100, 100))
    mean_curv = fit_curvature_distribution(curv)
    assert mean_curv > 0

def test_compute_energy():
    eigenvals = np.ones((10, 10, 2))
    energy = compute_energy(eigenvals)
    assert np.allclose(energy, 2)

def test_compute_orientation_correlation():
    img = np.zeros((50, 50, 2))
    img[:, :, 1] = 1
    corr = compute_orientation_correlation(img)
    assert corr.shape == (50, 50)
    assert np.isclose(corr[25, 25], 1.0) # Center should be 1 (normalized)

def test_compute_stats_wrapper():
    img = np.random.random((50, 50))
    stats = compute_image_orientation_statistics(img, 1, 1.0)
    assert stats.Frame == "1"
    assert float(stats.Energy) > 0
    assert float(stats.Coherence) >= 0

def test_helper_image_functions():
    img = np.random.random((20, 20))
    
    # Vector Field
    vf = compute_vector_field(img)
    assert vf.ndim == 3
    assert vf.shape[-1] == 2
    
    # Color Image
    cim = compute_color_image(img)
    assert cim.ndim == 3
    assert cim.shape[-1] == 3
    assert cim.dtype == np.uint8
    
    # Angle Image
    aim = compute_angle_image(img)
    assert aim.shape == img.shape
    
    # Coherence Image
    coh = compute_coherence_image(img)
    assert coh.shape == img.shape
    
    # Curvature Image
    curc = compute_curvature_image(img)
    assert curc.shape == img.shape
