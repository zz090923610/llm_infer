import subprocess
import unittest

import numpy as np

from py.generate import generate, generate_batch
from py.gguf_loader import load_model
from py.model import LlamaModel
from py.tests.conftest import LLAMA_BIN, MODEL
from py.tokenizer import Tokenizer, apply_chat_template

PROMPT = apply_chat_template([{"role": "user", "content": "Say hi in one word."}])
N_TOKENS = 8


@unittest.skipUnless(MODEL.is_file(), f"missing {MODEL}")
class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded = load_model(MODEL, progress=True)
        cls.model = LlamaModel(loaded)
        cls.tok = Tokenizer.from_gguf(loaded.gguf, loaded.hparams)

    def test_greedy_smoke(self):
        ids = self.tok.encode(PROMPT, parse_special=True)
        cache = self.model.new_cache(batch=1, max_seq=512)
        logits = self.model.forward(np.asarray(ids, dtype=np.int32)[None, :], cache)
        top5 = np.argsort(logits[0])[-5:][::-1]
        TestGenerate._top5 = [int(i) for i in top5]
        got = [int(np.argmax(logits[0]))]
        if got[0] != self.model.hparams.eos_id:
            for tid in generate(
                self.model,
                self.tok,
                [got[0]],
                max_tokens=N_TOKENS - 1,
                temperature=0.0,
                cache=cache,
            ):
                got.append(tid)
                if tid == self.model.hparams.eos_id:
                    break
        self.assertTrue(got)
        text = self.tok.decode(got, skip_special=True)
        self.assertTrue(text.strip())
        TestGenerate._greedy_ids = got
        TestGenerate._greedy_text = text

    def test_padded_batch_runs(self):
        prompts = [
            apply_chat_template([{"role": "user", "content": "Hi"}]),
            apply_chat_template([{"role": "user", "content": "2+2="}]),
        ]
        outs = generate_batch(self.model, self.tok, prompts, max_tokens=2, temperature=0.0)
        self.assertEqual(len(outs), 2)
        self.assertTrue(all(len(x) >= 1 for x in outs))

    @unittest.skipUnless((LLAMA_BIN / "llama-completion").is_file(), "llama-completion missing")
    def test_greedy_close_to_llama_cpp(self):
        """ggml kernels vs NumPy f32 can flip near-ties; require llama's first token in our top-5."""
        if not getattr(TestGenerate, "_greedy_ids", None):
            self.test_greedy_smoke()
        proc = subprocess.run(
            [
                str(LLAMA_BIN / "llama-completion"),
                "-m",
                str(MODEL),
                "-p",
                PROMPT,
                "-n",
                str(N_TOKENS),
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1.0",
                "--min-p",
                "0.0",
                "--repeat-penalty",
                "1.0",
                "-no-cnv",
                "--no-jinja",
                "--no-display-prompt",
                "-c",
                "512",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        ref = proc.stdout.strip().split("llama_perf")[0].strip()
        self.assertTrue(ref)
        first = self.tok.encode(ref, parse_special=False)[0]
        self.assertIn(
            first,
            TestGenerate._top5,
            f"llama first token {first} {self.tok.vocab[first]!r} not in our top-5 "
            f"{[(i, self.tok.vocab[i]) for i in TestGenerate._top5]}; ours={TestGenerate._greedy_text!r} ref={ref!r}",
        )


if __name__ == "__main__":
    unittest.main()
