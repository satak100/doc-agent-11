from PIL import Image

from doc_agent.contracts import Page
from doc_agent.ingest.preprocess import enhance_image
from doc_agent.ingest.loader import load_pages
from doc_agent.vision.layout import detect


def test_ingest_is_stable_and_ignores_non_images(tmp_path):
    Image.new("RGB", (10, 20), "white").save(tmp_path / "b.jpg")
    Image.new("RGB", (20, 10), "white").save(tmp_path / "a.png")
    (tmp_path / "note.txt").write_text("ignore")
    pages = load_pages({"paths": {"images": str(tmp_path)}})
    assert [page.doc_id for page in pages] == ["a.png", "b.jpg"]
    assert len({page.id for page in pages}) == 2


def test_preprocess_and_layout_preserve_page_geometry(tmp_path):
    path = tmp_path / "page.jpg"
    Image.new("RGB", (100, 200), "white").save(path)
    with Image.open(path) as image:
        assert enhance_image(image).size == (100, 200)
    page = Page(id="p", image_path=str(path), doc_id="page.jpg")
    regions = detect([page], {"layout": {"heading_ratio": 0.2}})
    assert [(r.kind, r.bbox) for r in regions] == [
        ("heading", (0, 0, 100, 40)),
        ("text", (0, 40, 100, 200)),
    ]
