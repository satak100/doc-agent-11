from doc_agent.vision.ocr import _clean


def test_ocr_cleanup_removes_locations_not_unicode():
    raw = "Tab. Phoscon<|LOC_1|><|LOC_2|>\nওষুধ<|LOC_3|>"
    assert _clean(raw) == "Tab. Phoscon\nওষুধ"
