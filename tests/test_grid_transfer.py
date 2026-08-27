import numpy as np

from pssolver.control import conservative_block_average, periodic_translate_axis


def test_block_average_preserves_constant_and_cell_average_coordinates():
    shape = (8, 6, 4)
    axes = [np.arange(size, dtype=float) + 0.5 for size in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    values = 2.0 * x - 3.0 * y + 0.5 * z

    restricted = conservative_block_average(values, (2, 3, 2))
    coarse_axes = [
        (np.arange(count, dtype=float) + 0.5) * factor
        for count, factor in zip(restricted.shape, (2, 3, 2))
    ]
    cx, cy, cz = np.meshgrid(*coarse_axes, indexing="ij")
    np.testing.assert_allclose(restricted, 2.0 * cx - 3.0 * cy + 0.5 * cz)


def test_block_average_preserves_trailing_q_components_and_mean():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(8, 6, 4, 5)).astype(np.float32)
    restricted = conservative_block_average(values, (2, 3, 2))
    assert restricted.shape == (4, 2, 2, 5)
    np.testing.assert_allclose(
        restricted.mean(axis=(0, 1, 2)),
        values.mean(axis=(0, 1, 2)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_periodic_translation_matches_integer_roll_and_is_reversible():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(32, 4, 3, 5)).astype(np.float32)
    spacing = 8.0 / values.shape[0]
    displacement = -5 * spacing
    translated = periodic_translate_axis(values, displacement, 8.0)
    np.testing.assert_allclose(translated, np.roll(values, -5, axis=0), atol=2e-6)
    restored = periodic_translate_axis(translated, -displacement, 8.0)
    np.testing.assert_allclose(restored, values, atol=3e-6)
