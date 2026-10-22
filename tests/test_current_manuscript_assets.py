from pathlib import Path

ROOT = Path(__file__).parents[1]
PAIRS = [
    'fig_latent_template_stress.png',
    'fig_cgh_probe_agreement.png',
    'fig_methylation_heldout.png',
]

def test_current_result_figures_match_reader_facing_copies():
    for name in PAIRS:
        paper = ROOT / 'docs/manuscripts/shared/figures/results' / name
        website = ROOT / 'docs/assets/figures' / name
        assert paper.is_file() and website.is_file()
        assert paper.read_bytes() == website.read_bytes()

def test_one_authoritative_paper_and_book():
    doc = ROOT / 'docs/manuscripts'
    paper = doc / 'paper/main.tex'
    assert paper.is_file() and (doc / 'book/main.tex').is_file()
    assert 'onecolumn' in paper.read_text(encoding='utf-8')
    assert not (doc / 'executive').exists()
    assert not (ROOT / 'docs/revision').exists()
