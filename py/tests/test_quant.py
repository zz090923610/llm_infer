import unittest

import numpy as np

from py.quant import BLOCK_Q8_0, QK8_0, dequantize_q8_0


class TestQ80(unittest.TestCase):
    def test_known_block(self):
        scale = np.float16(0.5)
        qs = np.arange(QK8_0, dtype=np.int8) - 16
        block = scale.tobytes() + qs.tobytes()
        self.assertEqual(len(block), BLOCK_Q8_0)
        y = dequantize_q8_0(block, QK8_0)
        expected = qs.astype(np.float32) * np.float32(0.5)
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)

    def test_two_blocks_independent_scales(self):
        d0 = np.float16(1.0).tobytes()
        d1 = np.float16(2.0).tobytes()
        q0 = np.ones(QK8_0, dtype=np.int8).tobytes()
        q1 = np.full(QK8_0, 3, dtype=np.int8).tobytes()
        y = dequantize_q8_0(d0 + q0 + d1 + q1, QK8_0 * 2)
        np.testing.assert_allclose(y[:32], 1.0)
        np.testing.assert_allclose(y[32:], 6.0)

    def test_rejects_ragged_length(self):
        with self.assertRaises(ValueError):
            dequantize_q8_0(b"\x00" * 10, 10)


if __name__ == "__main__":
    unittest.main()
