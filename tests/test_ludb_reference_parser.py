
from types import SimpleNamespace

from scripts.benchmark_ludb_landmarks import qrs_triplets


def test_qrs_parser_ignores_p_and_t():

    ann = SimpleNamespace(
        symbol=[
            "(", "p", ")",
            "(", "N", ")",
            "(", "t", ")",
            "(", "N", ")",
        ],
        sample=[
            1, 2, 3,
            10, 15, 20,
            25, 30, 35,
            40, 45, 50,
        ],
    )

    assert qrs_triplets(ann) == [
        (10, 15, 20),
        (40, 45, 50),
    ]


def test_qrs_parser_rejects_invalid_order():

    ann = SimpleNamespace(
        symbol=["(", "N", ")"],
        sample=[10, 9, 20],
    )

    assert qrs_triplets(ann) == []
