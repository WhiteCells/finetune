from __future__ import annotations

import unittest

try:
    from model.loader import model_dtype_kwarg_name
    from model.loader import resize_token_embeddings_if_needed
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers"}:
        raise
    model_dtype_kwarg_name = None  # type: ignore[assignment]
    resize_token_embeddings_if_needed = None  # type: ignore[assignment]


class FakeEmbeddings:
    def __init__(self, num_embeddings: int) -> None:
        self.num_embeddings = num_embeddings


class FakeModel:
    def __init__(self, num_embeddings: int) -> None:
        self.embeddings = FakeEmbeddings(num_embeddings)
        self.resize_calls: list[int] = []

    def get_input_embeddings(self) -> FakeEmbeddings:
        return self.embeddings

    def resize_token_embeddings(self, tokenizer_size: int) -> None:
        self.resize_calls.append(tokenizer_size)
        self.embeddings.num_embeddings = tokenizer_size


@unittest.skipUnless(resize_token_embeddings_if_needed is not None, "需要安装项目依赖")
class ModelLoaderTests(unittest.TestCase):
    @unittest.skipUnless(model_dtype_kwarg_name is not None, "需要安装项目依赖")
    def test_uses_current_transformers_dtype_kwarg(self) -> None:
        self.assertIn(model_dtype_kwarg_name(), {"dtype", "torch_dtype"})

    def test_does_not_shrink_reserved_embedding_slots(self) -> None:
        model = FakeModel(num_embeddings=151936)

        resize_token_embeddings_if_needed(model=model, tokenizer_size=151669)

        self.assertEqual(model.resize_calls, [])
        self.assertEqual(model.embeddings.num_embeddings, 151936)

    def test_grows_embeddings_when_tokenizer_is_larger(self) -> None:
        model = FakeModel(num_embeddings=100)

        resize_token_embeddings_if_needed(model=model, tokenizer_size=120)

        self.assertEqual(model.resize_calls, [120])
        self.assertEqual(model.embeddings.num_embeddings, 120)


if __name__ == "__main__":
    unittest.main()
