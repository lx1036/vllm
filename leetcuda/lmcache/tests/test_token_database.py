import pytest

from leetcuda.lmcache.config import LMCacheEngineConfig
from leetcuda.lmcache.tests.utils import create_engine_metadata, generate_tokens
from leetcuda.lmcache.token_database import ChunkedTokenDatabase

import torch

@pytest.mark.parametrize('chunk_length', [16, 64, 256])
def test_chunked_token_database(chunk_length):
    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")

    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_length, backend="cpu")
    metadata = create_engine_metadata()
    db = ChunkedTokenDatabase(cfg, metadata)
    # Process without mask
    original_results = list(db.process_tokens(tokens))
    for i in range(0, test_length, chunk_length):
        st, ed, key = original_results[i // chunk_length] # // 向下取整
        assert st == i
        assert ed == min(i + chunk_length, test_length)

    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")
    num_falses = [
        i * chunk_length for i in range(0, test_length // chunk_length)
    ]
    for i in range(0, test_length // chunk_length):
        mask[:num_falses[i]] = False
        new_results = list(db.process_tokens(tokens, mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]
