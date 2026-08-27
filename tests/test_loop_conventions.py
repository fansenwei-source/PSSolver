import numpy as np

from pssolver.loop_conventions import (
    classify_ideal_loop,
    convention_metadata,
    ideal_loop_axis_angles,
    validate_nml_frame,
)


N = np.array((1.0, 0.0, 0.0))
M = np.array((0.0, 1.0, 0.0))
L = np.array((0.0, 0.0, 1.0))
AXES_NML = np.column_stack((N, M, L))


def test_yingyou_ideal_loop_mode_mapping():
    expected = ((N, "pure-splay"), (M, "pure-bend"), (L, "pure-twist"))
    for k, loop_type in expected:
        measured_type, angles = classify_ideal_loop(k, AXES_NML)
        assert measured_type == loop_type
        assert angles[loop_type] < 1e-10


def test_yingyou_metadata_separates_principal_and_loop_planes():
    metadata = convention_metadata(N, M, L, k=N, nu=N, omega=L)
    assert metadata["classified_type"] == "pure-splay"
    assert metadata["principal_plane"] == "N-M"
    assert metadata["principal_plane_normal"] == "L"
    assert metadata["loop_plane_normal"] == "nu"
    assert metadata["angle_k_nu_deg"] < 1e-10
    assert abs(metadata["gamma_angle_nu_Omega_deg"] - 90.0) < 1e-10


def test_nml_frame_must_be_right_handed_and_orthonormal():
    np.testing.assert_allclose(validate_nml_frame(N, M, L), AXES_NML)
    try:
        validate_nml_frame(N, M, -L)
    except ValueError as error:
        assert "right-handed" in str(error)
    else:
        raise AssertionError("A left-handed NML frame was accepted.")


def test_ideal_loop_axis_angles_are_nematic():
    positive = ideal_loop_axis_angles(N, AXES_NML)
    negative = ideal_loop_axis_angles(-N, AXES_NML)
    assert positive == negative
