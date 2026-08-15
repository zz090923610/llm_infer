import json
import subprocess
import unittest

from py.tests.conftest import LLAMA_BIN, MODEL
from py.tokenizer import Tokenizer, apply_chat_template


def _llama_ids(text: str) -> list[int]:
    exe = LLAMA_BIN / "llama-tokenize"
    proc = subprocess.run(
        [str(exe), "-m", str(MODEL), "-p", text, "--ids", "--no-bos"],
        check=True,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


@unittest.skipUnless(MODEL.is_file(), f"missing {MODEL}")
class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.from_file(str(MODEL))

    def test_roundtrip_ascii(self):
        text = "Hello, world! 123 cats."
        ids = self.tok.encode(text, parse_special=False)
        self.assertEqual(self.tok.decode(ids, skip_special=False), text)

    def test_stop_ids_include_end_marker(self):
        stops = self.tok.stop_ids
        self.assertIn(self.tok.hparams.eos_id, stops)
        self.assertIn(self.tok.token_to_id["END"], stops)
        self.assertIn(self.tok.token_to_id["ĠEND"], stops)
        prompt = apply_chat_template([{"role": "user", "content": "Hi"}])
        ids = self.tok.encode(prompt, parse_special=True)
        self.assertEqual(ids[0], self.tok.token_to_id["<|im_start|>"])
        self.assertIn(self.tok.token_to_id["<|im_end|>"], ids)

    @unittest.skipUnless((LLAMA_BIN / "llama-tokenize").is_file(), "llama-tokenize missing")
    def test_matches_llama_tokenize(self):
        samples = [
            "Hello world",
            "The number 42 is nice.",
            apply_chat_template([{"role": "user", "content": "Say hi"}]),
            "It's a test — café 123\n\nNext.",
        ]
        for text in samples:
            ours = self.tok.encode(text, parse_special=True)
            ref = _llama_ids(text)
            self.assertEqual(ours, ref, f"mismatch on {text!r}\n ours={ours}\n ref={ref}")


if __name__ == "__main__":
    unittest.main()
