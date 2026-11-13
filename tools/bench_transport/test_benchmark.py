"""Benchmark-local correctness checks, independent of the production CPU suite."""
import numpy as np
import pytest
from common import RawSerializer, canonical, payload

@pytest.mark.parametrize('dtype', ['<f4', '>f4', 'uint8', 'int16'])
def test_raw_roundtrip_preserves_dtype_shape_values_for_strided_arrays(dtype):
    values = np.arange(60, dtype=dtype).reshape(10, 6)[::2, ::2]
    restored = RawSerializer.from_bytes(RawSerializer.to_bytes({'state': values}))['state']
    assert restored.dtype == values.dtype and restored.shape == values.shape
    np.testing.assert_array_equal(restored, values)


def test_rejects_object_arrays_and_preserves_empty_arrays():
    with pytest.raises(ValueError):
        RawSerializer.to_bytes(np.array([object()], dtype=object))
    empty = np.empty((0, 6), dtype=np.float32)
    restored = RawSerializer.from_bytes(RawSerializer.to_bytes(empty))
    assert restored.shape == empty.shape and restored.dtype == empty.dtype


def test_payloads_are_seeded_and_keep_camera_order():
    a, b = payload(3, 224, 224), payload(3, 224, 224)
    for key in a['images']:
        np.testing.assert_array_equal(a['images'][key], b['images'][key])
    assert list(canonical(a)) == ['observation.state', *[f'observation.images.cam{i}' for i in range(3)]]
    assert a['state'].shape == (6,) and a['state'].dtype == np.float32
