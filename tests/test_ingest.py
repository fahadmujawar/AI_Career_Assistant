import pytest

from rag.ingest import chunk_text


def make_text(n):
    return " ".join(f"w{i}" for i in range(n))


def test_chunks_overlap():
    chunks = chunk_text(make_text(90), chunk_size=50, overlap=10)

    assert len(chunks) == 2
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_short_last_piece_is_dropped():
    # Windows start at 0, 40, 80; the last one has only 15 words (< 20).
    chunks = chunk_text(make_text(95), chunk_size=50, overlap=10)

    assert len(chunks) == 2
    assert chunks[-1].split()[-1] == "w89"


def test_text_under_20_words_gives_no_chunks():
    assert chunk_text(make_text(15), chunk_size=50, overlap=10) == []


def test_default_sizes_200_words_gives_2_chunks():
    chunks = chunk_text(make_text(200), chunk_size=150, overlap=30)

    assert len(chunks) == 2
    assert len(chunks[0].split()) == 150
    assert len(chunks[1].split()) == 80


@pytest.mark.parametrize("overlap", [50, 60])
def test_overlap_not_smaller_than_chunk_size_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text(make_text(100), chunk_size=50, overlap=overlap)
